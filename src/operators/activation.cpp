//
// Created by sun on 2021/12/13.
//

#include"Ops.h"

Sigmoid::Sigmoid(TensorHandle _input,Graph* g):OpBase(_input,OP_SIGMOID,g){
    auto t = new Tensor,t1 = new Tensor,t2 = new Tensor,t3 = new Tensor;
    graph->push(new Neg(t,_input,graph->num_cpt));
    graph->push(new Exp(t1,t,graph->num_cpt));
    graph->push(new Add(t2,t1,&Tensor::one,graph->num_cpt));
    graph->push(new Rec(t3,t2,graph->num_cpt));
    numOutputs = 1;
    outputs[0] = t3;
}


Softmax::Softmax(TensorHandle _input, const Variable& v,Graph *g):OpBase(_input,OP_SOFTMAX,g) {
    auto t = new Tensor,t1 = new Tensor,t2 = new Tensor,t3 = new Tensor;
    graph->push(new Exp(t,_input,graph->num_cpt));
    graph->push(new Sum(t1,t,{v},graph->num_cpt));
    graph->push(new Rec(t2,t1,graph->num_cpt));
    graph->push(new Mul(t3,t,t2,graph->num_cpt));
    outputs[0]  = t3;
}

Tanh::Tanh(TensorHandle _input,Graph* g):OpBase(_input,OP_TANH,g){
    /*
     *
     */
    auto t = new Tensor,t1 = new Tensor,t2 = new Tensor,t3 = new Tensor,t4 = new Tensor,t5 = new Tensor,t6 = new Tensor,t7=new Tensor;
    graph->push(new Exp(t,_input,graph->num_cpt));//e^x
    graph->push(new Neg(t1,_input,graph->num_cpt));
    graph->push(new Exp(t2,t1,graph->num_cpt)); //e^-x
    graph->push(new Neg(t3,t2,graph->num_cpt));//-e^-x
    graph->push(new Add(t4,t,t3,graph->num_cpt));// numerator
    graph->push(new Add(t5,t,t2,graph->num_cpt));//denominator
    graph->push(new Rec(t6,t5,graph->num_cpt));
    graph->push(new Mul(t7,t4,t6,graph->num_cpt));
    outputs[0] = t7;
}