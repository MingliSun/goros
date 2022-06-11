//
// Created by sun on 2021/12/31.
//

#include"Ops.h"
#include"Pattern.h"
#include"Producer.h"
Split::Split(TensorHandle _input,const vector<int>& sections,int axis,Graph* g)
    :OpBase(_input,OP_SPLIT,g),sections(sections){
    // not check
    Variable& v = _input->shape[axis];
    numOutputs = int(this->sections.size()+1);
    this->sections.insert(sections.begin(),v.get_lower_bound());
    this->sections.insert(sections.end(),v.get_upper_bound());
    auto arr = new Tensor[numOutputs];
    int length = _input->shape.size();
    for(int i=0;i<numOutputs;i++){
        auto new_v = graph->get_variable(axis,sections[i+1]-sections[i],length);
        auto out_shape = PatternMatch::shape_replace_with_variable(_input->shape,v,new_v);
        map<Variable,Affine> m = {make_pair(v,Parser::make_affine({new_v.name,"+",sections[i]}))};
        graph->push(new Assign(arr+i,_input,out_shape,graph->num_cpt,m));// autogen formula
        outputs[i] = arr+i;
    }
}

