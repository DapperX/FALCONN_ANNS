#include <cstdio>
#include <cstdlib>
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

template<class U, class V>
void output_recall(LSHNearestNeighborTable<U,uint32_t> &tbl, parlay::internal::timer &t, uint32_t num_probe, uint32_t recall, 
	uint32_t cnt_query, std::vector<V> &q, std::vector<uint32_t*> &gt, uint32_t rank_max)
{
	//std::vector<std::vector<std::pair<uint32_t,float>>> res(cnt_query);
	parlay::sequence<std::vector<uint32_t>> res(cnt_query);
	/* // TODO
	parlay::parallel_for(0, cnt_query, [&](size_t i){
		res[i] = g.search(q[i], recall, num_probe);
	});
	*/
	t.next("Warm-up search");
	 // TODO
	auto query_object = tbl.construct_query_object(num_probe);

	// TODO
	/*
	parlay::parallel_for(0, cnt_query, [&](size_t i){
		// flag_query
		// search_control ctrl{};
		using PointType = typename to_densevec<float>::type;
		auto qobj = *dynamic_cast<

			falconn::wrapper::LSHNNQueryWrapper<
				PointType,
				uint32_t,
				float,
				falconn::core::StaticLSHTable<PointType, KeyType, LSHType, HashType,CompositeHashTableType, DataStorageType>,
				float,
				falconn::wrapper::PointTypeTraitsInternal<PointType>::CosineDistance,
				DataStorage
			>

			falconn::wrapper::LSHNNQueryWrapper<Eigen::Matrix<float, -1, 1>, int, float, falconn::core::StaticLSHTable<Eigen::Matrix<float, -1, 1>, unsigned int, falconn::core::CrossPolytopeHashDense<float, long unsigned int>, long unsigned int, falconn::core::StaticCompositeHashTable<long unsigned int, unsigned int, falconn::core::StaticLinearProbingHashTable<long unsigned int, unsigned int> >, falconn::core::ArrayDataStorage<Eigen::Matrix<float, -1, 1>, unsigned int> >, float, falconn::core::EuclideanDistanceDense<float>, falconn::core::ArrayDataStorage<Eigen::Matrix<float, -1, 1>, unsigned int> >*
		>(query_object.get());
		qobj.find_k_nearest_neighbors(q[i], recall, &res[i]);
	});
	*/

	for(size_t i=0; i<cnt_query; ++i)
	{
		// flag_query
		// search_control ctrl{};
		// query_object->find_k_nearest_neighbors(q[i], recall, &res[i]);
		query_object->get_candidates_with_duplicates(q[i], &res[i]);
	};
	
	//auto t2 = std::chrono::high_resolution_clock::now();
	double time_query = t.next_time();
	//printf("time diff: %.8f\n", time_query);
	// auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
	//std::chrono::duration<double, std::milli> diff = t2-t1;
	//printf("time diff (hi): %.4f\n", diff.count());
	// t.report(time_query, "Find neighbors");
	printf("FALCONN: Find neighbors: %.4f\n", time_query);

	if(rank_max<recall)
		recall = rank_max;
//	uint32_t cnt_all_shot = 0;
	std::vector<uint32_t> result(recall+1);
	printf("measure recall@%u with num_probe=%u on %u queries\n", recall, num_probe, cnt_query);
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
	uint32_t cnt_shot = 0;
	for(uint32_t i=0; i<=recall; ++i)
	{
		printf("%u ", result[i]);
		cnt_shot += result[i]*i;
	}
	putchar('\n');
	printf("%.6f at %ekqps\n", float(cnt_shot)/cnt_query/recall, cnt_query/time_query/1000);
	/* // TODO
	printf("# visited: %lu\n", g.total_visited.load());
	printf("# eval: %lu\n", g.total_eval.load());
	printf("size of C: %lu\n", g.total_size_C.load());
	*/
	auto statistics = query_object->get_query_statistics();
	cout << "average total query time: " << statistics.average_total_query_time << endl;
	cout << "average lsh time: " << statistics.average_lsh_time << endl;
	cout << "average hash table time: " << statistics.average_hash_table_time << endl;
	cout << "average distance time: " << statistics.average_distance_time << endl;
	cout << "average number of candidates: " << statistics.average_num_candidates << endl;
	cout << "average number of unique candidates: " << statistics.average_num_unique_candidates << endl;
	puts("---");
}

template<class U>
void output_recall(LSHNearestNeighborTable<U,uint32_t> &tbl, commandLine param, parlay::internal::timer &t)
{
	if(param.getOption("-?"))
	{
		printf(__func__);
		puts(
			"[-q <queryFile>] [-g <groundtruthFile>]"
			"-ef <ef_query> [-r <recall@R>=1] [-k <numQuery>=all]"
		);
		return;
	};
	char* file_query = param.getOptionValue("-q");
	char* file_groundtruth = param.getOptionValue("-g");
	auto [q,_] = load_point(file_query, to_point<float>); // TODO
	t.next("Read queryFile");

	uint32_t cnt_rank_cmp = param.getOptionIntValue("-r", 1);
//	const uint32_t ef = param.getOptionIntValue("-ef", cnt_rank_cmp*50);
	const uint32_t cnt_pts_query = param.getOptionIntValue("-k", q.size());

	auto [gt,rank_max] = load_point(file_groundtruth, gt_converter<uint32_t>{});
	for(uint32_t i=100; i<6000; i+=500)
		output_recall(tbl, t, i, cnt_rank_cmp, cnt_pts_query, q, gt, rank_max);
}

template<typename U>
void run_test(commandLine args) // intend to pass by value
{
	using type_point = typename to_densevec<U>::type;

	const char* file_in = args.getOptionValue("-in");
	const uint32_t cnt_points = args.getOptionLongValue("-n", 0);
	// const float m_l = args.getOptionDoubleValue("-ml", 0.36);
	const uint32_t num_hashtbl = args.getOptionIntValue("-l", 40);
	const uint32_t num_hashbit = args.getOptionIntValue("-b", 18);
	const uint32_t num_rotations = args.getOptionIntValue("-rot", 1);
	const auto lsh_family = args.getOptionValue("-lsh", "cp");
	const auto dist_func = args.getOptionValue("-dist", "L2");
	// const bool do_fixing = !!args.getOptionIntValue("-f", 0);
	// flag_query = args.getOptionIntValue("-flag", 0);

	parlay::internal::timer t("FALCONN", true);

	// using T = typename U::type_elem; // TODO
	auto [ps,dim] = load_point(file_in, to_point<U/*// TODO*/>, cnt_points);
	t.next("Read inFile");
	printf("col: %lu\n", ps[0].cols());
	printf("row: %lu\n", ps[0].rows());
	printf("size: %lu\n", ps[0].size());

	 // TODO
	falconn::LSHConstructionParameters params;
	// TODO: get default parameter
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
	falconn::compute_number_of_hash_functions<type_point>(num_hashbit, &params);
	params.num_setup_threads = 0; // use up all the threads
	params.storage_hash_table = falconn::StorageHashTable::BitPackedFlatHashTable;
	fputs("Start building FALCONN\n", stderr);
	auto ptbl = falconn::construct_table<type_point,uint32_t>(ps, params); // TODO
	t.next("Build index");

	/* // TODO
	size_t cnt_degree = g.cnt_degree();
	printf("total degree: %lu\n", cnt_degree);
	t.next("Count degrees");
	*/

	output_recall(*ptbl, args, t);
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-type <elemType> -dist <distance> -n <numInput> -ml <m_l> -m <m> "
		"-efc <ef_construction> -alpha <alpha> -r <recall@R> [-b <batchBase>]"
		"-in <inFile> ..."
	);

	/* // TODO
	const char* dist = parameter.getOptionValue("-dist");
	const char* type = parameter.getOptionValue("-type");
	*/
	run_test<float>(parameter);
	return 0;
}
