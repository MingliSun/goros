//
// Created by sun on 2022/1/6.
//

#ifndef AUTOFUSION_POLYNOMIAL_H
#define AUTOFUSION_POLYNOMIAL_H
#include<variant>
#include<string>
#include<vector>
#include<map>
#include<list>
#include <set>
#include"Affine.h"
using namespace std;

enum DataType {
    DT_FLOAT = 0, // float - fp32
    DT_DOUBLE = 1, //double  -fp64
    DT_HALF = 2, //_Float16 - fp16
    DT_INT8 = 3, //int8_t
    DT_UINT8 = 4, //uint8_t
    DT_INT16 = 5,//int16_t
    DT_UINT16, //uint16_t
    DT_INT32 ,//int32_t
    DT_UINT32,//uint32_t
    DT_INT64 ,//int64_t
    DT_UINT64,//uint64_t
    DT_BIT ,//bit
};


using base = std::variant<int,float>;
base operator+(const base&a,const base&b);
base operator*(const base&a,const base&b);
ostream& operator<<(ostream& os ,const base& value);

enum IntermediaType{
    INTERMEDIA_AFFINE,
    INTERMEDIA_POLYNOMIAL,
    INTERMEDIA_VARIABLE,
    INTERMEDIA_ELEMENT,// load from Tensor
    INTERMEDIA_CALL,
};

class Poly1d{
public:
    Poly1d() {
        coef[0] = 0;
    }
    explicit Poly1d(const vector<base>& coef);
    Poly1d operator*(const Poly1d& that) const;
    void update(int key,const base& value);
    float get_coef(int key);
private:
    map<int,base> coef;
};

enum AtomType{//
    ERROR=0,
    VARIABLE,
    SYMBOL,
    CONSTANT,////integer for affine
    END,
};
const vector<string> atom_type_string = {
        "error","variable","symbol","constant","end",
};
class Atom{
public:
    AtomType type{END};
    base constant{};// int or float
    string symbol;// keywords or identifier
    Atom() = default;
    explicit  Atom(AtomType t):type(t){}
    Atom(AtomType t,base c):type(t),constant(c){}
    Atom(AtomType t,string symbol):type(t),symbol(std::move(symbol)){}
};

/*
 *  base : variable constant(float) parameter(ignore)
 *  operation: + - * /(div) ~(negation) %(mod) floor ceil exp log tan cos sin sqrt sinh cosh abs power min max
 *  other operation: tanh sigmoid sinh cosh(not support yet)
 *  cause we support floor ... ,we can't use coefficient like Affine do, we use postfix data structure
 */
class Polynomial{
public:
    list<Atom> infix; // for print
    list<Atom> postfix;//data structure
    set<string> variables;
    float evaluate(const map<string,float>& m);
};

class Function;

class Intermedia{
public:
    string name;//guid
    DataType dataType{DT_FLOAT};
    IntermediaType intermediaType{};
    Affine affine;
    Polynomial polynomial;
    string var;
    pair<string,vector<string>> tensor_index;
    map<string,vector<string>> call; // name==>parameter
    Intermedia(string  name, const string& index);//load or variable
    Intermedia(string name,const vector<string>& v,const string& tensor_name);
    Intermedia(string  name,const Affine& v); //rarely used
    Intermedia(string  name,Polynomial  v,DataType t=DT_FLOAT);//polynomial
    Intermedia(string name,string func_name,vector<string> parameter); // call

    //todo
    base evaluate(const map<string,float>& m);
    [[nodiscard]] string to_string() const;
    bool operator<(const Intermedia& that) const{
        return this->name < that.name;
    }
};
class Function{//declare and definition
public:
    string name;
    vector<string> signature;
    set<Intermedia> procedure;
    Function(string name,vector<string> signature);
    bool operator<(const Function& that) const{
        return this->name < that.name;
    }
};

#endif //AUTOFUSION_POLYNOMIAL_H
