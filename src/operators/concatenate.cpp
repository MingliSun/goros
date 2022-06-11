//
// Created by sun on 2021/12/11.
//

#include"Ops.h"
#include<cassert>
#include<iostream>
Concatenate::Concatenate(const vector<TensorHandle>& inputs,int axis,Graph* g):OpBase(inputs,OP_CONCAT,g),axis(axis) {
    assert(inputs.size()>=2);
    assert(axis<inputs[0]->shape.size());
    Variable concat_axis = inputs[0]->shape[axis];

    auto first = inputs[0];
    TensorHandle second;
    for(int i=1;i<inputs.size();i++){
        auto t = new Tensor;
        second = inputs[i];
       // cout<<inputs[0]->to_concrete_shape_string()<<endl;
        //cout<<inputs[i]->to_concrete_shape_string()<<endl;
        assert(check(inputs[0]->shape,inputs[i]->shape));
        Variable v;
        int value = first->shape[axis].get_upper_bound() + second->shape[axis].get_upper_bound();
        if(concat_axis.name[0] == 'c') v = Graph::get_variable(graph->channel, value, 0, 1);
        else if(concat_axis.name[0]=='o') v = Graph::get_variable(graph->weight_oc, value,1,0);
        else if(concat_axis.name[0]=='n') v = Graph::get_variable(graph->input_n, value,0,0);
        else v = Graph::get_variable(graph->other, value,0,0);
        graph->push(new Cond(t,first,second,first->shape[axis],second->shape[axis],v,graph->num_cpt));
        first = second;
        second = t;
    }
    numOutputs = 1;
    outputs[0] = second;
}

bool Concatenate::check(const vector<Variable> &shape, const vector<Variable> &shape2) const {
    if(shape.size()!=shape2.size()) return false;
    for(int i=0;i<shape.size();i++){
        if(i==axis) continue;
        if(shape[i]!=shape2[i]) return false;
    }
    return true;
}
