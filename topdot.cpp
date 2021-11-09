#pragma GCC diagnostic ignored "-Wunused-result"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <queue>

using namespace std;

struct candidate {
    int index; 
    double value;
};

bool candidate_cmp(candidate c_i, candidate c_j) {
    return (c_i.value > c_j.value);
}

class sort_fn {
    public:
        bool operator()(pair<double, int> &a, pair<double, int> &b) { 
          return a.first > b.first;
        }
};

// void __topdot(
//         int n_row,
//         int n_col,
//         int* Ap, int* Aj, double* Ax,
//         int* Bp, int* Bj, double* Bx,
//         int k,
//         double lower_bound,
//         int* Cj, double* Cx
// ) {
    
//     #pragma omp parallel for
//     for(int row = 0; row < n_row; row++){
//         std::vector<int>    next(n_col, -1); // map instead of vector?
//         std::vector<double> sums(n_col,  0); // map instead of vector?
//         std::vector<candidate> candidates;
        
//         int head   = -2;
//         int length =  0;

//         // SpMM
//         int a_offset_start = Ap[row];
//         int a_offset_end   = Ap[row+1];
        
//         for(int a_col_idx = a_offset_start; a_col_idx < a_offset_end; a_col_idx++){
            
//             int a_col    = Aj[a_col_idx];
//             double a_val = Ax[a_col_idx];

//             int b_offset_start = Bp[a_col];
//             int b_offset_end   = Bp[a_col + 1];
            
//             for(int b_col_idx = b_offset_start; b_col_idx < b_offset_end; b_col_idx++){
                
//                 int b_col    = Bj[b_col_idx];
//                 double b_val = Bx[b_col_idx];
                
//                 sums[b_col] += a_val * b_val;
                
//                 // keep pointer to previous nonzero entry
//                 if(next[b_col] == -1){
//                     next[b_col] = head;
//                     head        = b_col;
//                     length++;
//                 }
//             }
//         }

//         // Collect results
//         int num_candidates = 0;
//         for(int i = 0; i < length; i++){
//             if(sums[head] > lower_bound){
//                 candidate c;
//                 c.index = head;
//                 c.value = sums[head];
//                 candidates.push_back(c);
//                 num_candidates++;
//             }
            
//             head = next[head];
//         }
        
//         // Get top-k
//         if (num_candidates > k){
//             std::nth_element(
//                 candidates.begin(),
//                 candidates.begin() + k,
//                 candidates.end(),
//                 candidate_cmp
//             );
//         }
        
//         for(int entry_idx = 0; entry_idx < k; entry_idx++){
//             if(entry_idx < num_candidates) {
//                 Cj[row * k + entry_idx] = candidates[entry_idx].index;
//                 Cx[row * k + entry_idx] = candidates[entry_idx].value;
//             } else {
//                 Cj[row * k + entry_idx] = -1;
//                 Cx[row * k + entry_idx] = -1;
//             }
//         }
//     }
//     return;
// }

void __topdot(
        int n_row,
        int n_col,
        int* Ap, int* Aj, double* Ax,
        int* Bp, int* Bj, double* Bx,
        int k,
        double lower_bound,
        int* Cj, double* Cx
) {

    int n_threads;
    #pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
    
    int* g_next    = (int*)malloc(n_threads * n_col * sizeof(int));
    double* g_sums = (double*)malloc(n_threads * n_col * sizeof(double));
    
    // #pragma omp parallel for
    for(int row = 0; row < n_row; row++){
        // std::vector<candidate> candidates;
        priority_queue<pair<double, int>, vector<pair<double, int>>, sort_fn> q;
        
        int tid      = omp_get_thread_num();
        int* next    = g_next + tid * n_col;
        double* sums = g_sums + tid * n_col;
        
        for(int i = 0; i < n_col; i++) next[i] = -1;
        for(int i = 0; i < n_col; i++) sums[i] = 0;
        
        int head   = -2;
        int length =  0;

        // SpMM
        int a_offset_start = Ap[row];
        int a_offset_end   = Ap[row+1];
        
        for(int a_col_idx = a_offset_start; a_col_idx < a_offset_end; a_col_idx++){
            
            int a_col    = Aj[a_col_idx];
            double a_val = Ax[a_col_idx];

            int b_offset_start = Bp[a_col];
            int b_offset_end   = Bp[a_col + 1];
            
            for(int b_col_idx = b_offset_start; b_col_idx < b_offset_end; b_col_idx++){
                
                int b_col    = Bj[b_col_idx];
                double b_val = Bx[b_col_idx];
                
                sums[b_col] += a_val * b_val;
                
                // keep pointer to previous nonzero entry
                if(next[b_col] == -1){
                    next[b_col] = head;
                    head        = b_col;
                    length++;
                }
            }
        }
        cout << length << endl;

        // Collect results
        for(int i = 0; i < k; i++) {
            if(i >= length) break;
            if(head == -2) break;
            
            if(sums[head] > lower_bound)
                q.push(make_pair(sums[head], head));
            
            head = next[head];
        }
        
        for(int i = k; i < length; i++) {
            if(i >= length) break;
            
            if(sums[head] > q.top().first) {
                q.push(make_pair(sums[head], head));
                q.pop();
            }
            head = next[head];
        }
        
        for(int i = 0; i < k; i++) {
            if(q.size() == 0) break;
            auto tmp = q.top();
            Cj[row * k + i] = tmp.second;
            Cx[row * k + i] = tmp.first;
            q.pop();
        }
    }
    return;
}

void __topdot2(
        int n_row,
        int n_col,
        int* A_indptr, int* A_indices, double* A_data,
        int* B_indptr, int* B_indices, double* B_data,
        int k,
        double lower_bound,
        int* Cj, double* Cx
) {
    
    int n_threads;
    #pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
    
    int* idxs_     = (int*)malloc(n_threads * n_col * sizeof(int));
    double* sums_  = (double*)malloc(n_threads * n_col * sizeof(double));
    
    #pragma omp parallel for
    for(int row = 0; row < n_row; row++){
        
        int tid      = omp_get_thread_num();
        int* idxs    = idxs_ + tid * n_col;
        double* sums = sums_ + tid * n_col;
        
        for(int i = 0; i < n_col; i++) idxs[i] = i;
        for(int i = 0; i < n_col; i++) sums[i] = 0;
        
        for(int a_offset = A_indptr[row]; a_offset < A_indptr[row + 1]; a_offset++){
            
            int a_col    = A_indices[a_offset];
            double a_val = A_data[a_offset];
            
            for(int b_offset = B_indptr[a_col]; b_offset < B_indptr[a_col + 1]; b_offset++){
                int b_col    = B_indices[b_offset];
                double b_val = B_data[b_offset];
                sums[b_col] += a_val * b_val;
            }
        }
        
        std::nth_element(
            idxs,
            idxs + k,
            idxs + n_col,
            [&sums](int left, int right) {
                return sums[left] > sums[right];
            }
        );
        
        for(int entry_idx = 0; entry_idx < k; entry_idx++){
            if(sums[idxs[entry_idx]] != 0) {
                Cj[row * k + entry_idx] = idxs[entry_idx];
                Cx[row * k + entry_idx] = sums[idxs[entry_idx]];
            } else {
                Cj[row * k + entry_idx] = -1;
                Cx[row * k + entry_idx] = -1;
            }
        }
    }
    
    free(idxs_);
    free(sums_);
    
    return;
}


void _topdot(  
  int n_row,
  int n_col,
  
  py::array_t<int> Ap,
  py::array_t<int> Aj,
  py::array_t<double> Ax,
  
  py::array_t<int> Bp,
  py::array_t<int> Bj,
  py::array_t<double> Bx,
  
  int k,
  double lower_bound,
  
  py::array_t<int> Cj, 
  py::array_t<double> Cx
) {
    __topdot(
      n_row,
      n_col,
      static_cast<int*>(Ap.request().ptr),
      static_cast<int*>(Aj.request().ptr),
      static_cast<double*>(Ax.request().ptr),
      static_cast<int*>(Bp.request().ptr),
      static_cast<int*>(Bj.request().ptr),
      static_cast<double*>(Bx.request().ptr),
      k,
      lower_bound,
      static_cast<int*>(Cj.request().ptr),
      static_cast<double*>(Cx.request().ptr)
    );
}

void _topdot2(  
  int n_row,
  int n_col,
  
  py::array_t<int> Ap,
  py::array_t<int> Aj,
  py::array_t<double> Ax,
  
  py::array_t<int> Bp,
  py::array_t<int> Bj,
  py::array_t<double> Bx,
  
  int k,
  double lower_bound,
  
  py::array_t<int> Cj, 
  py::array_t<double> Cx
) {
    __topdot2(
      n_row,
      n_col,
      static_cast<int*>(Ap.request().ptr),
      static_cast<int*>(Aj.request().ptr),
      static_cast<double*>(Ax.request().ptr),
      static_cast<int*>(Bp.request().ptr),
      static_cast<int*>(Bj.request().ptr),
      static_cast<double*>(Bx.request().ptr),
      k,
      lower_bound,
      static_cast<int*>(Cj.request().ptr),
      static_cast<double*>(Cx.request().ptr)
    );
}

PYBIND11_MODULE(topdot, m) {
    m.def(
      "_topdot",
      &_topdot,
      "_topdot",
      py::arg("n_row"),
      py::arg("n_col"),
      
      py::arg("Ap"),
      py::arg("Aj"),
      py::arg("Ax"),
      
      py::arg("Bp"),
      py::arg("Bj"),
      py::arg("Bx"),
      
      py::arg("k"),
      py::arg("lower_bound"),
      
      py::arg("Cj"),
      py::arg("Cx")
    );

    m.def(
      "_topdot2",
      &_topdot2,
      "_topdot2",
      py::arg("n_row"),
      py::arg("n_col"),
      
      py::arg("Ap"),
      py::arg("Aj"),
      py::arg("Ax"),
      
      py::arg("Bp"),
      py::arg("Bj"),
      py::arg("Bx"),
      
      py::arg("k"),
      py::arg("lower_bound"),
      
      py::arg("Cj"),
      py::arg("Cx")
    );
}