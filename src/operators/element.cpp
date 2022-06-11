//
// Created by sun on 2021/11/18.
//

#include"Ops.h"
#include<cassert>

Element::Element(TensorHandle left,TensorHandle right,OpType _type,Graph*g):OpBase(left,right,_type,g){
    //elementwise legal check
    assert(_type==OP_EW_ADD||OP_EW_ADD==OP_EW_MUL);
    assert(left->shape==right->shape);

    numOutputs = 1;

    // 0D Tensor do not need a variable
    auto t = new Tensor;
    if(_type==OP_EW_ADD){
        graph->push(new Add(t,left,right,graph->num_cpt));
    }else if(_type==OP_EW_MUL){
        graph->push(new Mul(t,left,right,graph->num_cpt));
    }
    outputs[0] = t;
}

Element::~Element()  = default;
