//
// Created by sun on 2021/12/27.
//

#include"Ops.h"

Dense::Dense(TensorHandle _input, TensorHandle _weight, Graph* g): OpBase(_input, OP_MATMUL, g){
    //normal dense (len=2)and batched dense(len=3)
    // assuming last element of _input and first element of _weight equal

    if(_input->shape.size()==2&&_weight->shape.size()==2){
        assert(_input->shape.back().get_upper_bound()==_weight->shape.front().get_upper_bound());
        //we only use _weight inside this function,
        //it doesn't matter if we reshape _weight
        vector<Variable> new_vars;
        new_vars.push_back(_input->shape[1]);
        if(_weight->shape[1]==_input->shape[1])
            new_vars.emplace_back( Producer::get_unique_variable_name(),_weight->shape[1].get_upper_bound());
        else new_vars.push_back(_weight->shape[1]);
        _weight->reshape(new_vars);
        auto t = new Tensor,t1 = new Tensor;
        graph->push(new Mul(t,_input,_weight,graph->num_cpt));
        auto sum = new Sum(t1,t,{_weight->shape[0]},graph->num_cpt);
        graph->adjust_sum(sum);
        graph->push(sum);
        outputs[0] = t1;
        return;
    }else if(_input->shape.size()==3&&_weight->shape.size()==3){//batched dense
        assert(_input->shape.front().get_upper_bound()==_weight->shape.front().get_upper_bound());
        assert(_input->shape.back().get_upper_bound()==_weight->shape[1].get_upper_bound());
        vector<Variable> new_vars;
        new_vars.push_back(_input->shape[0]);
        new_vars.push_back(_input->shape[2]);
        if(_weight->shape[2]==_input->shape[2])
            new_vars.emplace_back( Producer::get_unique_variable_name(),_weight->shape[2].get_upper_bound());
        else new_vars.push_back(_weight->shape[2]);
        _weight->reshape(new_vars);
        auto t = new Tensor,t1 = new Tensor;
        graph->push(new Mul(t,_input,_weight,graph->num_cpt));
        auto sum = new Sum(t1,t,{_weight->shape[1]},graph->num_cpt);
        graph->adjust_sum(sum);
        graph->push(sum);
        outputs[0] = t1;
        return;
    }
    //dense only support normal dense and batched dense
    assert(false);
}