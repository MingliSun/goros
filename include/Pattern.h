//
// Created by sun on 2021/11/18.
//

#ifndef AUTOFUSION_PATTERN_H
#define AUTOFUSION_PATTERN_H
#include<string>
#include"Ops.h"
#include<cassert>
#include<cstdlib>
using namespace std;

enum CondMode{
    CondMode_VALID,
    CondMode_SAME,
    CondMode_INVALID
};


class PatternMatch{
public:
    Graph* graph;
    explicit PatternMatch(Graph* g);
    bool add2add_left(int idx);
    bool mul2mul_left(int idx);
    bool mul2add_left(int idx);
    bool mul2sum_left(int idx);
    bool mul2assign_left(int idx);
    bool mul2assign_both(int idx);
    bool add2sum_both(int idx);
    bool add2assign_both(int idx);
    bool sum2mul_single(int idx);
    bool assign2assign_single(int idx);
    bool cond2mul_both(int idx);
    bool cond2assign_both(int idx);
    bool cond2sum_both(int idx);
    bool cond2smax_both(int idx);
    bool cond2mul_left(int idx);
    bool mul2smax_left(int idx);
    bool smax2assign_both(int idx);
    bool assign2smax_single(int idx);
    [[nodiscard]] bool any2nop(int idx); //deprecated
    bool reshape2assign_single(int idx);
    bool add_sum_mul_assign_left(int idx);//multiple layer
    bool cond_sum_mul_assign_left(int idx);//multiple layer
    [[nodiscard]] bool reverse_sum_mul_assign_both(int idx);//reverse multiple layer,deprecated
    bool reverse_both(int idx);//reverse multiple layer

    //sum2sum
    //cond2cond


    void optimize();
    void preprocess_weights();
    void partition() const;
    void partition(Computation* _computation,Computation* group,int &idx,bool group_new=false) const;
    vector<pair<string,int>> subst_history;

    static vector<Variable> shape_replace_with_variable(const vector<Variable>& shape,const Variable& src,const Variable& tgt);
    static vector<Variable> shape_replace_with_variable(const vector<Variable>& shape,const vector<Variable>& src,const vector<Variable>& tgt);
    //helper function
    static bool is_relevant(const vector<Variable>& reduce_axis, const vector<Variable>& shape);
    static bool is_dim_expand(const vector<Variable>& initial, const vector<Variable>& next);
    bool is_same_tensor(TensorHandle,TensorHandle);
    bool is_same_tensor_recur(TensorHandle,TensorHandle);
    static bool is_same_affines(const vector<Affine>& a, const vector<Affine>& b, map<string,string>&m);
    static bool is_same_formula(Formula* left,Formula* right,map<string,string>&m);
    CondMode get_cond_mode(TensorHandle,TensorHandle,const Variable&,const Variable&);
    static Compare compare_2_axis(const vector<Variable>&one,const vector<Variable>& two);
    static int get_contain_index(const vector<Variable>& shape, const Variable& v); // -1 means not contain
    static bool is_contain(const vector<Variable>& big,const vector<Variable>& small);
    static bool is_overlap(const vector<Variable>& big,const vector<Variable>& small);

    void preprocess_weights_recur(Computation*);
    static bool is_multiple_usage(Computation* computation);
    static void release(const vector<Computation*>& v);
    static void delete_t(const vector<Computation*>& v);
    void release_isolated_chain(TensorHandle t);

    map<TensorHandle,TensorHandle> same_tensor;//if we build sub-operator with no same items,we do not need to judge if the two is same
    //todo if this flag is true, we can do more aggressive add_sum_mul_assign_left (do not need sub_op_type are same nonlinear)
    bool is_better_cond2add;
private:
    static bool is_boundary(Computation* computation,bool last=false) ;
    void init_num_consumer(Computation* computation) const;

};

#endif //AUTOFUSION_PATTERN_H
