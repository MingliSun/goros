//
// Created by sun on 2022/1/3.
//

#include"Ops.h"

Pad::Pad(TensorHandle _input,const vector<int>& pad_axis,const vector<int>& pad_value,Graph* g):OpBase(_input,OP_WHERE,g){
    assert(!pad_axis.empty());
    assert(pad_axis.size()==pad_value.size()||pad_axis.size()*2==pad_value.size());
    auto t = new Tensor;
    int length = _input->shape.size();
    map<Variable,Affine> m;
    vector<int> out_size;
    out_size.resize(length);
    for(uint i=0;i<pad_axis.size();i++){
        Variable& v = _input->shape[pad_axis[i]];
        if(pad_axis.size()==pad_value.size()){
            out_size[pad_axis[i]] = v.get_upper_bound() + 2 * pad_value[i];
        }else{
            out_size[pad_axis[i]] = v.get_upper_bound() + pad_value[2 * i] + pad_value[2 * i + 1];
        }
    }
    for(int i=0;i<length;i++){
        if(out_size[i]==0) out_size[i] = _input->shape[i].get_upper_bound();
    }
    vector<Variable> out_shape = graph->get_variable(out_size);
    if(pad_axis.size()==pad_value.size()){
        for(uint i=0;i<pad_axis.size();i++){
            Variable& v = _input->shape[pad_axis[i]];
            m.insert(make_pair(v,Parser::make_affine({out_shape[pad_axis[i]].name,"-",pad_value[i]})));
        }
    }else{
        for(uint i=0;i<pad_axis.size();i++){
            Variable& v = _input->shape[pad_axis[i]];
            m.insert(make_pair(v,Parser::make_affine({out_shape[pad_axis[i*2]].name,"-",pad_value[2*i]})));
        }
    }
    graph->push(new Assign(t,_input,out_shape,graph->num_cpt,m));// autogen formula
    outputs[0] = t;
}