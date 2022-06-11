//
// Created by sun on 2021/11/19.
//

#include"Ops.h"
#include<cassert>
BatchNorm::BatchNorm(TensorHandle _input, TensorHandle _scale,
          TensorHandle _bias, TensorHandle _mean, TensorHandle _var,
          const float _epsilon,Graph*g):OpBase(_input, _scale, _bias, _mean, _var, OP_BATCH_NORM,g),epsilon(_epsilon){
    assert(_input->shape.size() == 4);
    numOutputs = 1;
    auto t = new Tensor,t1= new Tensor,t2= new Tensor,t3= new Tensor,t4= new Tensor,t5= new Tensor,t6= new Tensor,t7= new Tensor;
    auto eps = new Tensor(vector<Variable>(), WEIGHT, &epsilon, "eps");

    //out  = (input-mean)/sqrt(var+eps)*scale+bias
    graph->push(new Neg(t,_mean,graph->num_cpt));
    graph->push(new Add(t1,_input,t,graph->num_cpt));
    graph->push(new Add(t2,_var,eps,graph->num_cpt));
    graph->push(new Sqrt(t3,t2,graph->num_cpt));
    graph->push(new Rec(t4,t3,graph->num_cpt));
    graph->push(new Mul(t5,t1,t4,graph->num_cpt));
    graph->push(new Mul(t6,t5,_scale,graph->num_cpt));
    graph->push(new Add(t7,t6,_bias,graph->num_cpt));
    outputs[0] = t7;
}

BatchNorm::~BatchNorm() = default;
