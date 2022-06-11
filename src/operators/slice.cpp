//
// Created by sun on 2022/1/4.
//

#include"Ops.h"
Slice::Slice(TensorHandle _input,const vector<int>& starts,const vector<int>& ends,const vector<int>& axes,const vector<int>& steps,Graph* g)
    :OpBase(_input,OP_SLICE,g){
    assert(starts.size()==ends.size());
    assert(starts.size()==axes.size());
    assert(starts.size()==steps.size());
    int length = starts.size();
    vector<int> out_size;
    out_size.resize(_input->shape.size());
    for(int i=0;i<length;i++){
        out_size[axes[i]] = (ends[i]-starts[i])/steps[i];
    }
    for(int i=0;i<_input->shape.size();i++){
        if(out_size[i]==0) out_size[i] = _input->shape[i].get_upper_bound();
    }
    vector<Variable> out_shape = graph->get_variable(out_size);
    map<Variable,Affine> m;
    for(int i=0;i<length;i++){
        m.insert(make_pair(_input->shape[i],Parser::make_affine({out_shape[i].name,"*",steps[i],"+",starts[i]})));
    }
    auto t = new Tensor;
    graph->push(new Assign(t,_input,out_shape,graph->num_cpt,m, nullptr));//set formula as nullptr, user need to make sure affine is legal
    outputs[0] =t;
}