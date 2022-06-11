//
// Created by sun on 2022/1/19.
//

#include"Ops.h"

Transpose::Transpose(TensorHandle _input,const vector<int>& perm,Graph* g):OpBase(_input,OP_TRANSPOSE,g){
    assert(perm.size()==_input->shape.size());
    vector<int> output_size;
    for(int i:perm){
        assert(i<perm.size());
        output_size.push_back(_input->shape[i].get_upper_bound());
    }
    vector<Variable> output_shape = graph->get_variable(output_size);
    vector<Affine> lambda;
    auto t = new Tensor;
    lambda.reserve(perm.size());
    for(int i : perm){
        lambda.emplace_back(output_shape[i]);
    }
    graph->push(new Assign(t,_input,output_shape,graph->num_cpt,lambda, nullptr));
    outputs[0] = t;

}