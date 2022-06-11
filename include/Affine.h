//
// Created by sun on 2021/12/16.
//

#ifndef AFFINE_AFFINE_H
#define AFFINE_AFFINE_H
#include<vector>
#include<string>
#include<map>
#include<list>
#include<set>
using namespace std;
//enum AffineOp{
//    AFFINE_ADD,
//    AFFINE_MUL,
//    AFFINE_SUB,//syntax sugar
//    AFFINE_DIV,//floordiv
//    AFFINE_MOD,//modulo
//    AFFINE_NEG,//
//};
enum AffineType{
    AFFINE_CONSTANT,
    AFFINE_VARIABLE,
    AFFINE_PARAMETER,
    AFFINE_EXPR,
};
//const vector<std::string> affine_op_string={
//        "+","*","-","/","%","~"
//};
[[maybe_unused]] const vector<std::string> affine_type_string={
        "constant","variable","parameter","expr"
};

class Bound{
public:
    int value{0};
    string bound_name;
    bool is_concrete{true};
    Bound():value(0),is_concrete(true) {

    };
    explicit Bound(int _value){
        value = _value;
    }
    explicit  Bound(string name):is_concrete(false){
        bound_name = std::move(name);
    }
    bool operator<(const Bound& bound) const{
        if(is_concrete) return value< bound.value;
        else return bound_name< bound.bound_name;
    }
    [[nodiscard]] string to_string() const;
};

class Variable{
public:
    static bool check_name(const string& name);
    string name;// global unique identifier
    Bound lower_bound;
    Bound upper_bound;
    Variable() = default;
    explicit Variable(const string& name);
    Variable(const string& name,int _upper,int _lower=0);
    Variable(const string& name,string _upper,string _lower);
    Variable(const Variable& v);
    bool operator!=(const Variable & v)const;
    bool operator==(const Variable& v)const;
    bool operator<(const Variable& v) const;

    [[maybe_unused]] [[nodiscard]] bool lt_bound(const Variable& v) const;
    [[nodiscard]] string to_string() const;

    [[nodiscard]] int get_upper_bound() const;
    [[nodiscard]] int get_lower_bound() const;

private:

};
//todo other special case to explore
class Affine {
private:
    map<string,int> coefficients;
    //// for divider
    map<pair<uint,Affine>,int> divider;

    map<pair<uint,Affine>,int>::iterator iterator;

    //helper function
    Affine normalize(int& factor);
    int get_denominator();
    bool is_special_case();
    static bool is_opposite(Affine a, Affine b);
    static bool is_same_string(const string&a,const string& b,map<string,string>&m);
public:

    AffineType type{AFFINE_EXPR};
    bool is_zero();
    set<string> variables;
    Affine() = default;
    explicit Affine(uint value);//constant
    explicit Affine(const Variable& v);
    Affine(const Affine& that);
    void update(const string& key, int val);
    void update(const pair<int,Affine>& key, int val);

    [[maybe_unused]] void erase(const string& key);
    void erase(const pair<int,Affine>& key);
    //Affine* (Affine::* func)(Affine*);
    void merge(const Affine& that,int multiple=1);
    //todo Affine* ,should free *that or memory leaks happened.
    // change parameter into (const Affine&) and return Affine -done
    [[nodiscard]] Affine add(const Affine& that) const;
    [[nodiscard]] Affine sub(const Affine& that) const;
    [[nodiscard]] Affine mul(const Affine& that) const;
    [[nodiscard]] Affine neg() const;
    Affine ceil();
    Affine div(const Affine& that); //change this
    Affine mod(const Affine& that) ;//chang this
    void replace(const string& source,const Affine& target);
    static bool is_same_pattern(const Affine&left,const Affine&right,map<string,string>& m);
    [[nodiscard]] string to_string(bool te=false) const;
    [[nodiscard]] bool check(const map<string,int>& eval_vap) const;
    int evaluate(const  map<string,int>& eval_vap) const;
    //todo: check if map is qualified
    //todo :evaluation with map
    //todo compare two affine expression is equal or not?

    bool operator<(const Affine& that)const;
    bool operator==(const Affine& that)const ;
    bool operator!=(const Affine& that)const ;

    bool is_pure_divider(Affine & affine,uint& num);
    bool is_coefficients_equals(int value=1);
};


#endif //AFFINE_AFFINE_H