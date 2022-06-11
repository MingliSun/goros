//
// Created by sun on 2021/12/31.
//

#include"layer.h"

/*
 * Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
 * large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
 */

TensorHandle get_feature(Graph* graph,TensorHandle input,const vector<int>& layers,const vector<int>& filters,bool batch_norm=false){
    /*
     * Get VGG feature body as stacks of convolutions.
     */
    auto internal_layer = input;
    for(uint i=0;i<layers.size();i++){
        for(int j=0;j<layers[j];j++){
            string name = to_string(i+1)+"_"+to_string(j+1);
            internal_layer = conv2d(graph,internal_layer,filters[i],3,1,PD_MODE_SAME,"conv"+name);
            internal_layer = bias_add(graph,internal_layer,"conv"+name+"_bias");
            if(batch_norm) internal_layer = batch_norm_infer(graph,internal_layer,0.01,"bn"+name);
            internal_layer = graph->relu(internal_layer);
        }
        internal_layer = graph->max_pool2d(internal_layer,2,2,2,2,PD_MODE_SAME);
    }
    return internal_layer;
}

TensorHandle get_classifier(Graph* graph,TensorHandle input,int num_classes){
    /*
     * Get VGG classifier layers as fc layers.
     */
    auto flatten = graph->batch_flatten(input);
    auto fc6 = dense_add_bias(graph,flatten,4096,"fc6");
    auto relu6 = graph->relu(fc6);
    auto drop6 = graph->dropout(relu6,0.5);
    auto fc7 = dense_add_bias(graph,drop6,4096,"fc7");
    auto relu7 = graph->relu(fc7);
    auto drop7 = graph->dropout(relu7,0.5);
    auto fc8 = dense_add_bias(graph,drop7,num_classes,"fc8");
    return fc8;
}

Graph* vgg(int batch_size, const vector<int>& image_shape,int  num_classes,int num_layers=11,bool batch_norm=false){
    vector<int> layers,filters = {64,128,256,512,512};
    switch (num_layers) {
        case 11:
            layers = {1,1,2,2,2};
            break;
        case 13:
            layers = {2,2,2,2,2};
            break;
        case 16:
            layers = {2,2,3,3,3};
            break;
        case 19:
            layers = {2,2,4,4,4};
            break;
        default:
            throw exception();
    }
    auto graph = new Graph;
    vector<int> data_shape = image_shape;
    data_shape.insert(data_shape.begin(),batch_size);
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    auto feature = get_feature(graph,data,layers,filters,batch_norm);
    auto classifier = get_classifier(graph,feature,num_classes);
    auto symbol = graph->softmax(classifier);
    graph->function({symbol});
    return graph;

}

void test_vgg(){
    int num_layers = 11;
    auto graph = vgg(1,{3,224,224},1000,num_layers,false);
    string name = string(__FUNCTION__ )+::to_string(num_layers);
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(name+".dot");
    graph->codegen_te(name+".py");
}