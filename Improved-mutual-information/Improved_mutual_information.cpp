/**********************************************************************************************
/* a new measure to quantify algorithm performance for classification and community detection
/* Implemented by WangZeJun
/* mail: wangzejunscut@126.com
**********************************************************************************************/

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unordered_set>
#include <cmath>

void splitString(const std::string& s, const std::string& a,
                     std::vector<std::string>& res)
{
	std::string::size_type pos1, pos2;
	pos1 = 0;
	pos2 = s.find(a);

    while (pos2 != s.npos)
    {
        res.push_back(s.substr(pos1, pos2 - pos1));
        pos1 = pos2 + a.length();
        pos2 = s.find(a, pos1);
    }

    if (pos1 != s.length())
    {
        res.push_back(s.substr(pos1));
    }
}

int main(int argc, char** argv)
{
	std::vector<std::string> args(argv, argv + argc);
	if (args.size() < 2)
	{
		std::cout << "missing input file." << std::endl;
		exit(EXIT_FAILURE);
	}

	// open input file
	std::string input(args[1]);
	if (input == "")
	{
		throw std::invalid_argument(
            input + " cannot be used for evaluating!");
	}

	std::ifstream ifs(input);
	if (!ifs.is_open())
	{
		throw std::invalid_argument(
			input + " cannot be opened for evaluating!");
	}

	// read label data from input file
	std::string line;
	std::vector<int> label1;
	std::vector<int> label2;
	std::unordered_set<int> set1;
	std::unordered_set<int> set2;
	while (std::getline(ifs, line))
	{
		std::vector<std::string> labels;
		splitString(line, " ", labels);
		int l1 = std::stoi(labels[0]);
		int l2 = std::stoi(labels[1]);
		label1.push_back(l1);
		label2.push_back(l2);
		set1.insert(l1);
		set2.insert(l2);
	}

	int n = label1.size();
	int R = set1.size();
	int S = set2.size();
	std::cout << "Read " << n << " objects with R = " << R << " and S = " << S << std::endl;

	// construct the contingency table
	std::vector<std::vector<int> > contingency_table(R, std::vector<int>(S));
	for (int i = 0; i < label1.size(); i++)
	{
		contingency_table[label1[i]][label2[i]]++;
	}
	std::vector<int> row_sum(R);
	std::vector<int> col_sum(S);
	int sum = 0;
	for (int i = 0; i < contingency_table.size(); i++)
	{
		sum = 0;
		for (int j = 0; j < contingency_table[0].size(); j++)
		{
			sum += contingency_table[i][j];
		}
		row_sum[i] = sum;
	}
	for (int i = 0; i < contingency_table[0].size(); i++)
	{
		sum = 0;
		for (int j = 0; j < contingency_table.size(); j++)
		{
			sum += contingency_table[j][i];
		}
		col_sum[i] = sum;
	}

	// print contingency_table info
	std::cout << std::endl << "Contingency table:" << std::endl;
	for (int i = 0; i < contingency_table.size(); i++)
	{
		std::cout << "[ ";
		for (int j = 0; j < contingency_table[0].size(); j++)
		{
			std::cout << contingency_table[i][j] << " ";
		}
		std::cout << "]" << std::endl;
	}
	std::cout << "Row sums:" << std::endl;
	std::cout << "[ ";
	for (int i = 0; i < row_sum.size(); i++)
	{
		std::cout << row_sum[i] << " ";
	}
	std::cout << "]" << std::endl;
	std::cout << "Col sums:" << std::endl;
	std::cout << "[ ";
	for (int i = 0; i < col_sum.size(); i++)
	{
		std::cout << col_sum[i] << " ";
	}
	std::cout << "]" << std::endl;

	// calculate the standard mutual information
	double I = lgamma(n + 1);
	for (int i = 0; i < contingency_table.size(); i++)
	{
		for (int j = 0; j < contingency_table[0].size(); j++)
		{
			I += lgamma(contingency_table[i][j] + 1);
		}
	}
	for (int i = 0; i < row_sum.size(); i++)
	{
		I -= lgamma(row_sum[i] + 1);
	}
	for (int i = 0; i < col_sum.size(); i++)
	{
		I -= lgamma(col_sum[i] + 1);
	}
	I /= (n * log(2));
	std::cout << std::endl << "Mutual information I = " << I << " bits per object." << std::endl;

	// calculate the correction
	double w = n / (n + 0.5 * R * S);
	std::vector<double> x(R);
	double x_sum = 0;
	double x_log_sum = 0;
	double temp = (1 - w) / R;
	for (int i = 0; i < x.size(); i++)
	{
		x[i] = temp + w * row_sum[i] / n;
		x_sum += x[i] * x[i];
		x_log_sum += log(x[i]);
	}
	std::vector<double> y(S);
	double y_sum = 0;
	double y_log_sum = 0;
	temp = (1 - w) / S;
	for (int i = 0; i < y.size(); i++)
	{
		y[i] = temp + w * col_sum[i] / n;
		y_sum += y[i] * y[i];
		y_log_sum += log(y[i]);
	}
	double nu = (double)((S + 1) / (S * x_sum)) - (double)(1.0 / S);
	double mu = (double)((R + 1) / (R * y_sum)) - (double)(1.0 / R);

	/* 
	std::cout << "w = " << w << std::endl;
	std::cout << "x_sum = " << x_sum << std::endl;
	std::cout << "y_sum = " << y_sum << std::endl;
	std::cout << "(S + 1) / (S * x_sum) = " << (S + 1) / (S * x_sum) << std::endl;
	std::cout << "(R + 1) / (R * y_sum) = " << (R + 1) / (R * y_sum) << std::endl;
	std::cout << "nu = " << nu << std::endl;
	std::cout << "mu = " << mu << std::endl;
	std::cout << "(R - 1) * (S - 1) * log(n + 0.5 * R * S) = " << (R - 1) * (S - 1) * log(n + 0.5 * R * S) << std::endl;
	std::cout << "0.5 * (R + nu - 2) * y_log_sum = " << 0.5 * (R + nu - 2) * y_log_sum << std::endl;
	std::cout << "0.5 * (S + mu - 2) * x_log_sum = " << 0.5 * (S + mu - 2) * x_log_sum << std::endl;
	std::cout << "0.5 * (lgamma(mu * R) + lgamma(nu * S)) = " << 0.5 * (lgamma(mu * R) + lgamma(nu * S)) << std::endl;
	std::cout << "R * (lgamma(S) + lgamma(mu)) = " << R * (lgamma(S) + lgamma(mu)) << std::endl;
	std::cout << "S * (lgamma(R) + lgamma(nu)) = " << S * (lgamma(R) + lgamma(nu)) << std::endl;
	*/

	double logOmega = (R - 1) * (S - 1) * log(n + 0.5 * R * S) + 0.5 * (R + nu - 2) * y_log_sum + \
						0.5 * (S + mu - 2) * x_log_sum + 0.5 * (lgamma(mu * R) + lgamma(nu * S) \
						- R * (lgamma(S) + lgamma(mu)) - S * (lgamma(R) + lgamma(nu)));

	std::cout << std::endl << "Estimated number of contingency tables Omega = " << exp(logOmega) << std::endl;

	// calculate the reduced mutual information
	std::cout << std::endl << "Reduced mutual information: M = " << I - logOmega / (n * log(2)) << " bits per object." << std::endl;

}
