#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "common/data.hpp"

using namespace std;

// In Linux, enable access to the basename function for POSIX-compatible system.
// This helps shortening the names to the instance files when logging to the CSV file.
#ifdef __linux__
#include <libgen.h>
#else
const char *basename(const char *path) {
   return path;
}
#endif

// Simple structure for computing statistical metrics.
struct StatMetrics {
   const string m_name;
   vector <double> m_data;
   double m_min, m_max, m_mean, m_sd;
   StatMetrics(const string &name): m_name(name) {}
   StatMetrics &add(double val);
   StatMetrics &computeMetrics();
   const string csvHeaders() const;
   const string csvData() const;
};

int main(int argc, char **argv) {
   if (argc != 3) {
      cout << "Usage: " << argv[0] << " <1:inst.path> <2:inst.fmt>\n";
      return EXIT_FAILURE;
   }

   cout << "--- Feature extractor for MathPDP ---\n";
   cout << "Instance path: " << argv[1] << "\n";
   cout << "Instance format: " << argv[2] << "\n";
   cout << endl;

   Data data;
   if (data.read_from_file(argv[1], argv[2]) == Data::READ_FAIL) {
      cout << "ERROR: Failed to read data from the file " << argv[1] << ".\n";
      return EXIT_FAILURE;
   }

   // The original code supports computing conflicts between requests through
   // a preprocessing routine. Hopefully we can extract data from this procedures
   // as features, so I will run the preprocessing.
   Data::pre_process(data);
   Data::compute_conflicts(data);
   cout << "Pre-processing: " << data.nref << " tightenings | " << data.nconf << " conflicts\n";

   // Statistical metrics.
   StatMetrics demands("demand");
   StatMetrics twBeg("tw.begin"), twEnd("tw.ending"), twDur("tw.duration");
   StatMetrics dist("distance");

   // Extract node demands and tw.
   for (const Node &n: data.nodes) {
      // Only consider the demands for pickup nodes.
      // They are symmetrics for the delivery nodes, i.e., same value but 
      // with negative. Also ignore nodes with no demand (the depot).
      if (n.demand > 0) {
         demands.add(n.demand);
      }

      // Ignores the depot information regarding time-windows.
      if (n.demand != 0) {
         twBeg.add(n.etw);
         twEnd.add(n.ltw);
         twDur.add(n.ltw - n.etw);
      }
   }

   // Computes metrics of the distance matrix.
   // Also compute a metric of how assymetric the matrix is.
   double asymDist = 0.0;
   for (size_t i = 0; i < data.nodes.size(); ++i) {
      for (size_t j = i+1; j < data.nodes.size(); ++j) {
         dist.add(data.times[i][j]);
         asymDist += pow(data.times[i][j] - data.times[j][i], 2);
      }
   }

   // Prepare the output file with the features.
   fstream csv("features.csv", ios::out | ios::app);
   if (csv.tellp() == 0) {
      csv 
         << "instance,"
         << "vehicle.cap,"
         << demands.csvHeaders() << ","
         << twBeg.csvHeaders() << ","
         << twEnd.csvHeaders() << ","
         << twDur.csvHeaders() << ","
         << dist.csvHeaders() << ","
         << "distance.asym.sq" << ","


         << "tightenings,"
         << "conflicts"
         << endl;
   }

   // Triggers the computation of the metrics.
   demands.computeMetrics();
   twBeg.computeMetrics();
   twEnd.computeMetrics();
   twDur.computeMetrics();
   dist.computeMetrics();

   // Extract and store the features in the CSV file.
   csv 
      << basename(argv[1]) << ","
      << data.vcap << ","
      << demands.csvData() << ","
      << twBeg.csvData() << ","
      << twEnd.csvData() << ","
      << twDur.csvData() << ","
      << dist.csvData() << ","
      << asymDist << ","


      << data.nref << ","
      << data.nconf
      << endl;

   return EXIT_SUCCESS;
}

StatMetrics &StatMetrics::add(double val) {
   m_data.push_back(val);
}

StatMetrics &StatMetrics::computeMetrics() {
   if (m_data.empty()) abort();

   // Starts by sorting the input data. This allows
   // running some of the queries in O(1) afterwards.
   sort(begin(m_data), end(m_data));

   m_min = m_data.front();
   m_max = m_data.back();

   // Compute mean.
   double meanSum = accumulate(begin(m_data), end(m_data), 0);
   m_mean = meanSum / static_cast<double>(m_data.size());

   // Compute standard deviation.
   if (m_data.size() > 1) {
      double sqDiff = .0;
      for (double d: m_data)
         sqDiff += pow(d - m_mean, 2);
      m_sd = sqrt(sqDiff/static_cast<double>(m_data.size()-1));
   } else {
      m_sd = .0;
   }
}

const string StatMetrics::csvHeaders() const {
   stringstream fmt;
   fmt 
      << m_name << ".min,"
      << m_name << ".max,"
      << m_name << ".count,"
      << m_name << ".mean,"
      << m_name << ".sd"
      ;
   return fmt.str();
}

const string StatMetrics::csvData() const {
   stringstream out;
   out 
      << m_min << ","
      << m_max << ","
      << m_data.size() << ","
      << m_mean << ","
      << m_sd 
   ;
   return out.str();
}

