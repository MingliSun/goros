//
// Created by sun on 2022/1/19.
//

/*
 * Adapted from TASO
 */

#include"layer.h"

TensorHandle attention(Graph* graph,TensorHandle input,int heads){
    int d_model = input->shape[1].get_upper_bound();
    int d_k = d_model / heads;
    assert(input->shape[1].get_upper_bound() % heads == 0);
    TensorHandle weights[3];
    for (auto & weight : weights) {
        weight = graph->new_weight({ d_model, d_model });
    }
    // compute query, key, value tensors
    auto q = graph->dense(input, weights[0]);
    auto k = graph->dense(input, weights[1]);
    auto v = graph->dense(input, weights[2]);
    // reshape query, key, value to multiple heads
    q = graph->reshape(q, { -1, heads, d_k });
    k = graph->reshape(k, { -1, heads, d_k });
    v = graph->reshape(v, { -1, heads, d_k });
    // transpose query, key, value for batched dense
    q = graph->transpose(q, { 1, 0, 2 });
    k = graph->transpose(k, { 1, 2, 0 });
    v = graph->transpose(v, { 1, 0, 2 });
    // perform matrix multiplications
    auto logits = graph->dense(q, k);
    auto output = graph->dense(logits, v);
    // transpose the output back
    output = graph->transpose(output, { 1, 0, 2 });
    output = graph->reshape(output, { input->shape[0].get_upper_bound(), input->shape[1].get_upper_bound() });

    // a final linear layer
    auto linear = graph->new_weight( { d_model, d_model });
    output = graph->dense(output, linear);
    return output;
}

Graph* bert(){
    const int seq_length = 64;
    const int hidden_dims = 1024;
    auto graph = new Graph();
    auto inp = graph->new_input({ seq_length, hidden_dims });
    inp = graph->relu(inp);
    auto t = inp;
    for (int i = 0; i < 8; i++) {
        t = attention(graph, t, 16);
    }
    graph->function({t});
    return graph;
}

void test_bert_1(){
    const int seq_length = 64;
    const int hidden_dims = 1024;
    int heads = 16;
    auto graph = new Graph();
    auto input = graph->new_input({ seq_length, hidden_dims },DT_FLOAT,"data");


    int d_model = input->shape[1].get_upper_bound();
    int d_k = d_model / heads;
    TensorHandle weights[3];
    for (auto & weight : weights) {
        weight = graph->new_weight({ d_model, d_model });
    }

    input = graph->dense(input,weights[2]);

    auto q = graph->dense(input, weights[0]);
    auto k = graph->dense(input, weights[1]);

    q = graph->reshape(q, { -1, heads, d_k });
    k = graph->reshape(k, { -1, heads, d_k });

    q = graph->transpose(q, { 1, 0, 2 });
    k = graph->transpose(k, { 1, 2, 0 });

    auto logits = graph->dense(q, k);
    graph->function({logits});
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ )+".dot");
    graph->codegen_te(string(__FUNCTION__)+".py");
}

void test_bert(){

    const int seq_length = 64;
    const int hidden_dims = 1024;
    auto graph = new Graph();
    auto inp = graph->new_input({ seq_length, hidden_dims },DT_FLOAT,"data");
    inp = graph->relu(inp);
    auto t = inp;
    t = attention(graph, t, 16);

    graph->function({t});



    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ )+".dot");
    graph->codegen_te(string(__FUNCTION__)+".py");
}