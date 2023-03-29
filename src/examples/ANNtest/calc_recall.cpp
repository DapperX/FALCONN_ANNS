#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <string>
#include <map>
#include <chrono>
#include <stdexcept>
// #include <memory>
//#include <H5Cpp.h>
#include "dist.hpp"
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <falconn/lsh_nn_table.h>
using std::cout, std::endl;
using falconn::LSHNearestNeighborTable;

template<typename T>
class to_densevec
{
public:
	using type = falconn::DenseVector<T>;
	using type_elem = T;

	template<typename Iter>
	type operator()([[maybe_unused]] uint32_t id, Iter begin, [[maybe_unused]] Iter end)
	{
		using type_src = typename std::iterator_traits<Iter>::value_type;
		static_assert(std::is_convertible_v<type_src,T>, "Cannot convert to the target type");

		const uint32_t dim = std::distance(begin, end);
		type p(dim);
		for(uint32_t i=0; i<dim; ++i)
			p[i] = *(begin+i);
		return p;
	}
};

template<typename T>
to_densevec<T> to_point;

template<typename T>
class gt_converter{
public:
	using type = T*;
	using type_elem = T;

	template<typename Iter>
	type operator()([[maybe_unused]] uint32_t id, Iter begin, Iter end)
	{
		using type_src = typename std::iterator_traits<Iter>::value_type;
		static_assert(std::is_convertible_v<type_src,T>, "Cannot convert to the target type");

		const uint32_t n = std::distance(begin, end);

		T *gt = new T[n];
		for(uint32_t i=0; i<n; ++i)
			gt[i] = *(begin+i);
		return gt;
	}
};

// Visit all the vectors in the given 2D array of points
// This triggers the page fetching if the vectors are mmap-ed
template<class T>
void visit_point(const T &array, size_t dim0, size_t dim1)
{
	parlay::parallel_for(0, dim0, [&](size_t i){
		const auto &a = array[i];
		[[maybe_unused]] volatile auto elem = a[0];
		for(size_t j=1; j<dim1; ++j)
			elem = a[j];
	});
}

template<typename T, class U, class V>
size_t output_recall(LSHNearestNeighborTable<U,uint32_t> &tbl, parlay::internal::timer &t, uint32_t num_probes, uint32_t recall, 
	uint32_t cnt_query, std::vector<V> &q, std::vector<uint32_t*> &gt, uint32_t rank_max)
{
	typedef std::pair<uint32_t,float> pair;
	std::unique_ptr<falconn::LSHNearestNeighborQueryPool<U,uint32_t>> 
		query_pool(tbl.construct_query_pool(num_probes, 1000*recall/*TODO*/, parlay::num_workers()));
	// parlay::sequence<std::vector<uint32_t>> res_raw(parlay::num_workers());
	// parlay::sequence<parlay::sequence<pair>> res(cnt_query);
	parlay::sequence<std::vector<uint32_t>> res(cnt_query);

	auto do_query = [&](size_t i){
		query_pool->find_k_nearest_neighbors(q[i], recall, &res[i]);
		/*
		const auto tid = parlay::worker_id();
		query_pool->get_candidates_with_duplicates(q, res_raw[tid]);
		res[i] = parlay::tabulate(res_raw[tid].size(), [&](size_t j){
			const auto v = res_raw[tid][j];
			return pair{v, };
		});
		*/
	};

	parlay::parallel_for(0, cnt_query, [&](size_t i){
		do_query(i);
	});
	t.next("Warm-up search");

	parlay::parallel_for(0, cnt_query, [&](size_t i){
		do_query(i);
	});
	double time_query = t.next_time();
	printf("FALCONN: Find neighbors: %.4f\n", time_query);

	if(rank_max<recall)
		recall = rank_max;
//	uint32_t cnt_all_shot = 0;
	std::vector<uint32_t> result(recall+1);
	printf("measure recall@%u with num_probes=%u on %u queries\n", recall, num_probes, cnt_query);
	for(uint32_t i=0; i<cnt_query; ++i)
	{
		uint32_t cnt_shot = 0;
		for(uint32_t j=0; j<recall; ++j)
			if(std::find_if(res[i].begin(),res[i].end(),[&](const uint32_t p){
				return p==gt[i][j];}) != res[i].end())
			{
				cnt_shot++;
			}
		result[cnt_shot]++;
	}
	// printf("#all shot: %u (%.2f)\n", cnt_all_shot, float(cnt_all_shot)/cnt_query);
	uint32_t total_shot = 0;
	for(uint32_t i=0; i<=recall; ++i)
	{
		printf("%u ", result[i]);
		total_shot += result[i]*i;
	}
	putchar('\n');
	printf("%.6f at %ekqps\n", float(total_shot)/cnt_query/recall, cnt_query/time_query/1000);
	/* // TODO
	printf("# visited: %lu\n", g.total_visited.load());
	printf("# eval: %lu\n", g.total_eval.load());
	printf("size of C: %lu\n", g.total_size_C.load());
	*/
	falconn::QueryStatistics stats = query_pool->get_query_statistics();
	cout << "average total query time: " << stats.average_total_query_time << endl;
	cout << "average lsh time: " << stats.average_lsh_time << endl;
	cout << "average hash table time: " << stats.average_hash_table_time << endl;
	cout << "average distance time: " << stats.average_distance_time << endl;
	cout << "average number of candidates: " << stats.average_num_candidates << endl;
	cout << "average number of unique candidates: " << stats.average_num_unique_candidates << endl;
	puts("---");
	return total_shot;
}

template<typename T, class U>
void output_recall(LSHNearestNeighborTable<U,uint32_t> &tbl, commandLine param, parlay::internal::timer &t, uint32_t dim, uint32_t L)
{
	const char* file_query = param.getOptionValue("-q");
	const char* file_groundtruth = param.getOptionValue("-g");
	auto [q,_] = load_point(file_query, to_point<T>);
	t.next("Read queryFile");
	printf("%s: [%lu,%u]\n", file_query, q.size(), _);

	visit_point(q, q.size(), dim); // TODO: eliminate the passed dim
	t.next("Fetch query vectors");

	auto [gt,rank_max] = load_point(file_groundtruth, gt_converter<uint32_t>{});
	t.next("Read groundTruthFile");
	printf("%s: [%lu,%u]\n", file_groundtruth, gt.size(), rank_max);

	auto parse_array = [](const std::string &s, auto f){
		std::stringstream ss;
		ss << s;
		std::string current;
		std::vector<decltype(f((char*)NULL))> res;
		while(std::getline(ss, current, ','))
			res.push_back(f(current.c_str()));
		std::sort(res.begin(), res.end());
		return res;
	};
	const uint32_t cnt_query = param.getOptionIntValue("-k", q.size());
	auto cnt_rank_cmp = parse_array(param.getOptionValue("-r"), atoi);
	auto threshold = parse_array(param.getOptionValue("-th"), atof);

	auto get_best = [&](uint32_t k, uint32_t num_probes){
		return output_recall<T>(tbl, t, num_probes, k, cnt_query, q, gt, rank_max);
	};

	for(auto k : cnt_rank_cmp)
	{
		uint32_t l_last = L;
		for(auto t : threshold)
		{
			printf("searching for k=%u, th=%f\n", k, t);
			const size_t target = t*cnt_query*k;
			uint32_t l=l_last, r_limit=k*150;
			uint32_t r = l;
			bool found = false;
			while(true)
			{
				// auto [best_shot, best_beta] = get_best(k, r);
				if(get_best(k,r)>=target)
				{
					found = true;
					break;
				}
				if(r==r_limit) break;
				r = std::min(r*2, r_limit);
			}
			if(!found) break;
			while(r-l>l*0.05+1)
			{
				const auto mid = (l+r)/2;
				const auto best_shot = get_best(k,mid);
				if(best_shot>=target)
					r = mid;
				else
					l = mid;
			}
			l_last = l;
		}
	}
}

template<typename T>
void run_test(commandLine args) // intend to pass by value
{
	using type_point = typename to_densevec<T>::type;

	const char* file_in = args.getOptionValue("-in");
	const uint32_t cnt_points = args.getOptionLongValue("-n", 0);

	parlay::internal::timer t("FALCONN", true);
	auto [ps,dim] = load_point(file_in, to_point<T>, cnt_points);
	t.next("Read inFile");
	printf("col: %lu\n", ps[0].cols());
	printf("row: %lu\n", ps[0].rows());
	printf("size: %lu\n", ps[0].size());

	visit_point(ps, ps.size(), dim);
	t.next("Fetch input vectors");

	const uint32_t num_hashtbl = args.getOptionIntValue("-l", 50); // 40~80
	const uint32_t num_rotations = args.getOptionIntValue("-rot", 1);
	const auto lsh_family = args.getOptionValue("-lsh", "cp");
	const auto dist_func = args.getOptionValue("-dist", "L2");
	const uint32_t num_hashbit = args.getOptionIntValue("-b", ceil(log2(ps.size()))-2);
	const uint32_t K = args.getOptionIntValue("-K", 0);
	const uint32_t lastk = args.getOptionIntValue("-lastk", 0);
	printf("num_hashbit: %u\n", num_hashtbl);

	falconn::LSHConstructionParameters params;
	if(lsh_family=="cp")
		params.lsh_family = falconn::LSHFamily::CrossPolytope;
	else if(lsh_family=="hp")
		params.lsh_family = falconn::LSHFamily::Hyperplane;
	else throw std::invalid_argument("Unrecognized hash family");

	if(dist_func=="L2")
		params.distance_function = falconn::DistanceFunction::EuclideanSquared;
	else if(dist_func=="ndot")
		params.distance_function = falconn::DistanceFunction::NegativeInnerProduct;
	else throw std::invalid_argument("Unrecognized distance function");

	params.dimension = dim;
	params.l = num_hashtbl;
	params.num_rotations = num_rotations;

	// This function will set both `k` and `last_cp_dimension`
	falconn::compute_number_of_hash_functions<type_point>(num_hashbit, &params);
	if(K) params.k = K;
	if(lastk) params.last_cp_dimension = lastk;

	params.num_setup_threads = 0; // use up all the threads
	params.storage_hash_table = falconn::StorageHashTable::BitPackedFlatHashTable;

	puts("======= key params for building FALCONN ======");
	printf("params.lsh_family: %s\n", lsh_family.c_str());
	printf("params.l: %ld\n", params.l);
	printf("params.k: %ld\n", params.k);
	printf("params.last_cp_dimension: %ld\n", params.last_cp_dimension);
	printf("params.num_rotations: %ld\n", params.num_rotations);

	fputs("Start building FALCONN\n", stderr);
	auto ptbl = falconn::construct_table<type_point,uint32_t>(ps, params); // TODO
	t.next("Build index");

	output_recall<T>(*ptbl, args, t, dim, num_hashtbl);
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-type <elemType> -n <numInput> -r <recall@R>,... -th <threshold>,... "
		"-in <inFile> -q <queryFile> -g <groundtruthFile> [-k <numQuery>=all] "
		"-dist (L2|ndot) -lsh (cp|hp) -l <numHashTable> [-b <numHashBit>] [-rot <numRotation>] "
		"[-K numHashFunc] [-lastk <last_cp_dimension>]"
	);

	const char* type = parameter.getOptionValue("-type");
	if(!strcmp(type,"float"))
		run_test<float>(parameter);
	else throw std::invalid_argument("Unsupported element type");	
	return 0;
}
