//
// Created by sun on 2022/1/6.
//

#include"Ops.h"
#include"Producer.h"
Winograd::Winograd(TensorHandle _input, TensorHandle _weight, int tile_size,PaddingMode _padding, Graph*g)
    :OpBase(_input,OP_CONVOLUTION,g){
    /*
     * Conv2D Winograd implementation in NCHW layout.
     *  tile_size : The size of the tile to use for the Winograd filter
     *  e.g 2 for F(2x2,3x3)
     *      3 for F(4x4,3x3)
     * _weight in OIHW layout only supporting 3x3 kernel yet
     * only supporting stride=1 yet
     *
     */
    assert(_input->shape.size() == _weight->shape.size());
    int size = _input->shape.size();
    assert(size>= 3);
    assert(_input->shape[1] == _weight->shape[1]);
    vector<int> input_size;
    vector<int> output_size;
    vector<int> npart;
    int P=1;
    int length = size-2;
    for(int i=2;i<size;i++){
        input_size.push_back(_input->shape[i].get_upper_bound());
        assert(_weight->shape[i].get_upper_bound()==3);
    }


    int r=3,m = tile_size;
    int alpha = m+r-1;
    void* ptr_A = Producer::allocate_winograd_A<float>(interpolation_points[m],alpha,m);
    void* ptr_B = Producer::allocate_winograd_B<float>(interpolation_points[m],alpha);
    void* ptr_G = Producer::allocate_winograd_G<float>(interpolation_points[m],alpha,r);

    TensorHandle data_pad;
    vector<int> pad_axis,pad_value;
    if(_padding==PD_MODE_SAME){
        for(int i=0;i<length;i++){
            pad_axis.push_back(i+2);
            pad_value.push_back(1);
            output_size.push_back(input_size[i]);
            npart.push_back((output_size[i]+m-1)/m);
            P*= npart[i];
        }
        data_pad= graph->pad(_input,pad_axis,pad_value);
    }else{
        data_pad = _input;
        for(int i=0;i<length;i++){
            output_size.push_back(input_size[i]-2);
            npart.push_back((output_size[i]+m-1)/m);
            P*= npart[i];
        }
    }
    vector<int> denominator = npart;
    denominator.push_back(1);
    for(int i=length-1;i>=0;i--){
        denominator[i] *=denominator[i+1];
    }
    npart.insert(npart.begin(),1);
    //weight transform
    TensorHandle temp=_weight;
    vector<Variable> reduce_axis;
    for(int i=0;i<length;i++){
        auto t1 = new Tensor;
        string name = "eps"+to_string(alpha)+"_"+to_string(i);
        auto G = new Tensor({Variable(name,alpha),_weight->shape[i+2]},WEIGHT,DT_FLOAT,ptr_G,"winograd_G");//r --> alpha
        graph->push(new Mul(t1,temp,G,graph->num_cpt));
        temp = t1;
        reduce_axis.push_back(_weight->shape[i+2]);
    }
    auto kernel_pack = new Tensor;
    graph->push(new Sum(kernel_pack, temp, reduce_axis, graph->num_cpt, pad_value));//stride == pad_value == 1
    //pack data tile
    P *= _input->shape[0].get_upper_bound();
    vector<int> pack_size = {P,_weight->shape[1].get_upper_bound(),alpha,alpha};
    vector<Variable> pack_shape = Graph::get_variable(pack_size);
    auto input_tile = new Tensor;
    vector<Affine> lambda;
    lambda.push_back(Parser::make_affine({pack_shape[0].name,"/",denominator[0]}));
    lambda.emplace_back(pack_shape[1]);
    for(int i=0;i<length;i++){
        lambda.push_back(Parser::make_affine({pack_shape[0].name,"/",denominator[i+1],"%",npart[i+1],"*",m,"+",pack_shape[i+2].name}));
    }
    graph->push(new Assign(input_tile,data_pad,pack_shape,graph->num_cpt,lambda, nullptr));//set formula as nullptr
    //transform data
    reduce_axis.clear();
    temp = input_tile;
    for(int i=0;i<length;i++){
        auto t2 = new Tensor;
        string name = "eps"+to_string(alpha)+"_"+to_string(i);
        auto B = new Tensor({input_tile->shape[i+2],Variable(name,alpha)},WEIGHT,DT_FLOAT,ptr_B,"winograd_B");// alpha alpha
        auto mul = new Mul(t2,temp,B,graph->num_cpt);
        mul->simplify_const_tensor_indices = B->shape; ////auto_scheduler_simplify_const_tensor_indices
        graph->push(mul);
        temp = t2;
        reduce_axis.push_back(input_tile->shape[i+2]);
    }
    auto data_pack = new Tensor;
    graph->push(new Sum(data_pack,temp,reduce_axis,graph->num_cpt,pad_value,false));
    //do batch gemm
    auto t2 = new Tensor;
    graph->push(new Mul(t2, data_pack, kernel_pack, graph->num_cpt));
    auto bgemm = new Tensor;
    auto sum = new Sum(bgemm, t2, {_weight->shape[1]}, graph->num_cpt, pad_value,false);
    graph->adjust_sum(sum);
    graph->push(sum);

    //inverse transform
    reduce_axis.clear();
    temp = bgemm;
    for(int i=0;i<length;i++){
        auto t3 = new Tensor;
        string name = "v"+to_string(m)+"_"+to_string(i);
        auto A = new Tensor({bgemm->shape[i+2],Variable(name,m)},WEIGHT,DT_FLOAT,ptr_A,"winograd_A");//alpha m
        auto mul = new Mul(t3,temp,A,graph->num_cpt);
        mul->simplify_const_tensor_indices = A->shape; ////auto_scheduler_simplify_const_tensor_indices
        graph->push(mul);
        temp = t3;
        reduce_axis.push_back(bgemm->shape[i+2]);
    }
    auto inverse = new Tensor;
    graph->push(new Sum(inverse,temp,reduce_axis,graph->num_cpt,pad_value));

    //output
    output_size.insert(output_size.begin(),_weight->shape[0].get_upper_bound());
    output_size.insert(output_size.begin(),_input->shape[0].get_upper_bound());
    vector<Variable> output_shape = Graph::get_variable(output_size);
    vector<Affine> lambda1;
    vector<atomi> a = {output_shape[0].name, "*", denominator[0]};
    for(int i=0;i<length;i++){
        a.emplace_back("+");
        a.emplace_back(output_shape[i+2].name);
        a.emplace_back("/");
        a.emplace_back(m);
        a.emplace_back("*");
        a.emplace_back(denominator[i+1]);
    }
    lambda1.emplace_back(Parser::make_affine(a));
    lambda1.emplace_back(output_shape[1]);
    for(int i=0;i<length;i++){
        lambda1.emplace_back(Parser::make_affine({output_shape[i+2].name,"%",m}));
    }
    auto output_winograd = new Tensor;
    graph->push(new Assign(output_winograd,inverse,output_shape,graph->num_cpt,lambda1, nullptr));
    outputs[0] = output_winograd;
    //set concat axis
    string name = "eps"+to_string(alpha)+"_0";
    Variable cat(name,alpha);
    if(graph->concat_axis.find(cat)==graph->concat_axis.end()){
        graph->concat_axis.insert(cat);
    }
}