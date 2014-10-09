#include "utils.h"
#include <cstdlib>

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXXi;
using Eigen::MatrixXd;
using Eigen::VectorXi;

#include "Eigen/SVD"
using Eigen::JacobiSVD;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;

#include <cmath>
using std::exp;
using std::log;
using std::floor;
using std::tanh;
using std::sinh;
using std::cosh;
#ifdef __GXX_EXPERIMENTAL_CXX0X__
using std::lgamma;
using std::tgamma;
#endif

#include <cstdlib>
using std::rand;

#include <set>
using std::set;
using std::pair;

#include <algorithm>
using std::greater;
using std::sort;

#include <limits>
using std::numeric_limits;

#include <random>
using std::mt19937;
using std::normal_distribution;
using std::gamma_distribution;

MatrixXd MDLDA::signum(const MatrixXd& matrix) {
	return (matrix.array() > 0.).cast<double>() - (matrix.array() < 0.).cast<double>();
}



double MDLDA::gamma(double x) {
	if (x <= 0.0)
		throw Exception("Argument to gamma function must be positive.");
	return tgamma(x);
}



ArrayXXd MDLDA::gamma(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = tgamma(arr(i));

	return result;
}



double MDLDA::lngamma(double x) {
	if (x <= 0.0)
		throw Exception("Argument to gamma function must be positive.");
	return lgamma(x);
}



ArrayXXd MDLDA::lngamma(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = lgamma(arr(i));

	return result;
}



ArrayXXd MDLDA::digamma(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = digamma(arr(i));

	return result;
}



double MDLDA::polygamma(int n, double x) {
	if(n < 1)
		return digamma(x);
	return pow(-1, n + 1) * gamma(n + 1) * zeta(n + 1, x);
}



ArrayXXd MDLDA::polygamma(int n, const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = polygamma(n, arr(i));

	return result;
}



ArrayXXd MDLDA::tanh(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = std::tanh(arr(i));

	return result;
}



ArrayXXd MDLDA::cosh(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = std::cosh(arr(i));

	return result;
}



ArrayXXd MDLDA::sinh(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = std::sinh(arr(i));

	return result;
}



ArrayXXd MDLDA::sech(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = 1. / std::cosh(arr(i));

	return result;
}



Array<double, 1, Dynamic> MDLDA::logSumExp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().sum().log();
}



Array<double, 1, Dynamic> MDLDA::logMeanExp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().mean().log();
}



int MDLDA::sampleHistogram(const ArrayXd& histogram) {
	double r = sampleUniform() * histogram.sum();

	for(int k = 0; k < histogram.size(); ++k) {
		if(r < histogram[k])
			return k;
		r -= histogram[k];
	}

	throw Exception("Something went wrong while sampling from histogram.");
}



double MDLDA::sampleUniform() {
	// return a number from the interval [0, 1[
	return static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
}



ArrayXXd MDLDA::sampleNormal(int m, int n) {
 	static mt19937 gen(rand());

	normal_distribution<double> normal;
	ArrayXXd samples(m, n);

	for(int i = 0; i < samples.size(); ++i)
		samples(i) = normal(gen);

	return samples;
}



ArrayXXd MDLDA::sampleGamma(int m, int n, int k) {
	ArrayXXd samples = ArrayXXd::Zero(m, n);

	for(int i = 0; i < k; ++i)
		samples -= ArrayXXd::Random(m, n).abs().log();

	return samples;
}



ArrayXd MDLDA::sampleDirichlet(const ArrayXd& alpha) {
	static mt19937 gen(rand());

	ArrayXd sample(alpha.size());

	// sample from gamma distribution
	for(int i = 0; i < sample.size(); ++i) {
		gamma_distribution<double> distribution(alpha[i], 1.);
		sample[i] = distribution(gen);
	}

	return sample / sample.sum();
}



ArrayXXd MDLDA::sampleDirichlet(int m, int n, double alpha) {
	static mt19937 gen(rand());
	gamma_distribution<double> distribution(alpha, 1.);

	ArrayXXd sample(m, n);

	// sample from gamma distribution
	for(int i = 0; i < sample.size(); ++i)
		sample(i) = distribution(gen);

	return sample.rowwise() / sample.colwise().sum();
}



/**
 * Algorithm due to Knuth, 1969.
 */
ArrayXXi MDLDA::samplePoisson(int m, int n, double lambda) {
	ArrayXXi samples(m, n);
	double threshold = exp(-lambda);

	#pragma omp parallel for
	for(int i = 0; i < samples.size(); ++i) {
		double p = rand() / static_cast<double>(RAND_MAX);
		int k = 0;

		while(p > threshold) {
			p *= rand() / static_cast<double>(RAND_MAX);
			k += 1;
		}

		samples(i) = k;
	}

	return samples;
}



/**
 * Algorithm due to Knuth, 1969.
 */
ArrayXXi MDLDA::samplePoisson(const ArrayXXd& lambda) {
	ArrayXXi samples(lambda.rows(), lambda.cols());
	ArrayXXd threshold = (-lambda).exp();

	#pragma omp parallel for
	for(int i = 0; i < samples.size(); ++i) {
		double p = rand() / static_cast<double>(RAND_MAX);
		int k = 0;

		while(p > threshold(i)) {
			k += 1;
			p *= rand() / static_cast<double>(RAND_MAX);
		}

		samples(i) = k;
	}

	return samples;
}



ArrayXXi MDLDA::sampleBinomial(int w, int h, int n, double p) {
	ArrayXXi samples = ArrayXXi::Zero(w, h);

	#pragma omp parallel for
	for(int i = 0; i < samples.size(); ++i) {
		// very naive algorithm for generating binomial samples
		for(int k = 0; k < n; ++k)
			if(rand() / static_cast<double>(RAND_MAX) < p)
				samples(i) += 1; 
	}

	return samples;
}



ArrayXXi MDLDA::sampleBinomial(const ArrayXXi& n, const ArrayXXd& p) {
	if(n.rows() != p.rows() || n.cols() != p.cols())
		throw Exception("n and p must be of the same size.");

	ArrayXXi samples = ArrayXXi::Zero(n.rows(), n.cols());

	#pragma omp parallel for
	for(int i = 0; i < samples.size(); ++i) {
		// very naive algorithm for generating binomial samples
		for(int k = 0; k < n(i); ++k)
			if(rand() / static_cast<double>(RAND_MAX) < p(i))
				samples(i) += 1;
	}

	return samples;
}



set<int> MDLDA::randomSelect(int k, int n) {
	if(k > n)
		throw Exception("k must be smaller than n.");
	if(k < 0 || n < 0)
		throw Exception("n and k must be non-negative.");

	// TODO: a hash map could be more efficient
	set<int> indices;

	if(k <= n / 2) {
		for(int i = 0; i < k; ++i)
			while(indices.insert(rand() % n).second != true) {
				// repeat until insertion successful
			}
	} else {
		// fill set with all indices
		for(int i = 0; i < n; ++i)
			indices.insert(i);
		for(int i = 0; i < n - k; ++i)
			while(!indices.erase(rand() % n)) {
				// repeat until deletion successful
			}
	}

	return indices;
}



VectorXi MDLDA::argSort(const VectorXd& data) {
	// create pairs of values and indices
	vector<pair<double, int> > pairs(data.size());
	for(int i = 0; i < data.size(); ++i) {
		pairs[i].first = data[i];
		pairs[i].second = i;
	}

	// sort values in descending order
	sort(pairs.begin(), pairs.end(), greater<pair<double, int> >());

	// store indices
	VectorXi indices(data.size());
	for(int i = 0; i < data.size(); ++i)
		indices[pairs[i].second] = i;

	return indices;
}



MatrixXd MDLDA::covariance(const MatrixXd& data) {
	MatrixXd dataCentered = data.colwise() - data.rowwise().mean().eval();
	return dataCentered * dataCentered.transpose() / data.cols();
}



MatrixXd MDLDA::covariance(const MatrixXd& input, const MatrixXd& output) {
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs must be the same.");

	MatrixXd inputCentered = input.colwise() - input.rowwise().mean().eval();
	MatrixXd outputCentered = output.colwise() - output.rowwise().mean().eval();
	return inputCentered * outputCentered.transpose() / output.cols();
}



MatrixXd MDLDA::corrCoef(const MatrixXd& data) {
	MatrixXd C = covariance(data);
	VectorXd c = C.diagonal();
	return C.array() / (c * c.transpose()).array().sqrt();
}



MatrixXd MDLDA::normalize(const MatrixXd& matrix) {
	return matrix.array().rowwise() / matrix.colwise().norm().eval().array();
}



MatrixXd MDLDA::pInverse(const MatrixXd& matrix) {
	if(matrix.size() == 0)
		return matrix.transpose();

	JacobiSVD<MatrixXd> svd(matrix, ComputeThinU | ComputeThinV);

	VectorXd svInv = svd.singularValues();

	for(int i = 0; i < svInv.size(); ++i)
		if(svInv[i] > 1e-8)
			svInv[i] = 1. / svInv[i];

	return svd.matrixV() * svInv.asDiagonal() * svd.matrixU().transpose();
}



double MDLDA::logDetPD(const MatrixXd& matrix) {
	return 2. * matrix.llt().matrixLLT().diagonal().array().log().sum();
}



MatrixXd MDLDA::deleteRows(const MatrixXd& matrix, vector<int> indices) {
	MatrixXd result = ArrayXXd::Zero(matrix.rows() - indices.size(), matrix.cols());

	sort(indices.begin(), indices.end());

	unsigned int idx = 0;

	for(int i = 0; i < matrix.rows(); ++i) {
		if(idx < indices.size() && indices[idx] == i) {
			++idx;
			continue;
		}
		result.row(i - idx) = matrix.row(i);
	}

	return result;
}



MatrixXd MDLDA::deleteCols(const MatrixXd& matrix, vector<int> indices) {
	MatrixXd result = ArrayXXd::Zero(matrix.rows(), matrix.cols() - indices.size());

	sort(indices.begin(), indices.end());

	unsigned int idx = 0;

	for(int i = 0; i < matrix.cols(); ++i) {
		if(idx < indices.size() && indices[idx] == i) {
			++idx;
			continue;
		}
		result.col(i - idx) = matrix.col(i);
	}

	return result;
}
