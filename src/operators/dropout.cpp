//
// Created by sun on 2021/12/10.
//

#include"Ops.h"
#include<cassert>
#include"Producer.h"
Dropout::Dropout(TensorHandle _input, float _level, Graph *g): OpBase(_input,OP_DROPOUT,g) , prob(1.0 - _level){
    assert(!_input->shape.empty());
    numOutputs = 1;
    auto sample = new Tensor(_input->shape, WEIGHT, _input->dataType, Producer::allocate_bernoulli<float>(_input->shape, prob), "dropout");
    auto t = new Tensor,t1 = new Tensor,t2 = new Tensor;
    auto divider = new Tensor({}, WEIGHT, DT_FLOAT, &prob, "prob");
    graph->push(new Mul(t,_input,sample,graph->num_cpt));
    graph->push(new Rec(t1,divider,graph->num_cpt));
    graph->push(new Mul(t2,t,t1,graph->num_cpt));
    outputs[0] = t2;
}