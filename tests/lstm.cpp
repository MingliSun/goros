//
// Created by sun on 2022/1/3.
//

#include"layer.h"

void lstm_cell(Graph *graph,
               TensorHandle state_c,
               TensorHandle state_h,
               TensorHandle& next_c,
               TensorHandle& next_h,
               int num_hidden,
               int batch_size=1,
               const string& name=""
               ){
    auto inputs = graph->new_weight({batch_size,num_hidden},DT_FLOAT,name+"_inputs");
    auto i2h_weight = graph->new_weight({4*num_hidden,num_hidden},DT_FLOAT,name+"_i2h_weight");
    auto i2h_bias = graph->new_weight({4*num_hidden},DT_FLOAT,name+"_i2h_bias");
    auto h2h_weight = graph->new_weight({4*num_hidden,num_hidden},DT_FLOAT,name+"_h2h_weight");
    auto h2h_bias = graph->new_weight({4*num_hidden},DT_FLOAT,name+"_h2h_bias");
    auto i2h = graph->dense(inputs,i2h_weight);
    i2h = graph->bias_add(i2h,i2h_bias);
    auto h2h = graph->dense(state_c,h2h_weight);
    h2h = graph->bias_add(h2h,h2h_bias);
    auto gates = graph->add(i2h,h2h);
    vector<TensorHandle> out;
    graph->split(gates,4,1,out);
    // in_gate
    out[0] = graph->sigmoid(out[0]);
    //forget_gate
    out[1] = graph->sigmoid(out[1]);
    //in_transform
    out[2] = graph->tanh(out[2]);
    //out_gate
    out[3] = graph->sigmoid(out[3]);
    auto temp1 = graph->multiply(out[1],state_h);
    auto temp2 = graph->multiply(out[0],out[2]);
    next_c = graph->add(temp1,temp2);
    next_h = graph->multiply(out[3],graph->tanh(next_c));
}

Graph* lstm(int iterations,int num_hidden,int batch_size=1){
    auto graph = new Graph;
    auto state_c = graph->new_weight({batch_size,num_hidden},DT_FLOAT,"init_c"); //todo allocate zero
    auto state_h = graph->new_weight({batch_size,num_hidden},DT_FLOAT,"init_h");//todo allocate zero
    ////question : do we only take the last next_h, what about internal next_h?
    TensorHandle next_c,next_h;
    for(int i=0;i<iterations;i++){
        lstm_cell(graph,state_c,state_h,next_c,next_h,num_hidden,batch_size,"lstm"+::to_string(i));
        state_c = next_c;
        state_h = next_h;
    }
    graph->function({next_h});
    return graph;
}

void test_lstm(){
    auto graph = lstm(16,16);
    //...
}