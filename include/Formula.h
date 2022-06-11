//
// Created by sun on 2021/12/20.
//

#ifndef AFFINE3_FORMULA_H
#define AFFINE3_FORMULA_H


#include <variant>
#include<stack>
#include"Affine.h"
// Presburger Formula
/*
 * A Presburger Formula is a boolean combination of comparisons between (Quasi) Affine Expressions.
 * It is recursively constructed from the following elements:
 * Base :
 *  Comparision between quasi-affine expression whose result is a boolean constant(true or false)
 *
 *  Operation:
 *  conjunction disjunction not implies equal exist all
 *
 */
enum AffineConstraint{
    CONSTRAINT_LT,
    CONSTRAINT_LE,
    CONSTRAINT_GT,
    CONSTRAINT_GE,
    CONSTRAINT_EQ,
    CONSTRAINT_NE,
};
const vector<string> affine_constraint_string={
        "<","<=",">",">=","==","!="
};
enum LogicSymbol{
    LOGIC_DISJUNCTION,
    LOGIC_CONJUNCTION,
    LOGIC_IMPLICATION,// not used
    LOGIC_EQUIVALENCE,// not used
    LOGIC_NEGATION,
    QUANTIFIER_EXIST,//not supported
    QUANTIFIER_ALL,//not supported

    ENTITY_CONSTRAINT,// for expression tree
};
const vector<string> logic_symbol_string={
        "||","&&","->","<->","-|","∃","∀","c"
};


class Constraint {
public:
    Constraint():constraint(CONSTRAINT_LT){};
    Constraint(AffineConstraint,const Affine&,const Affine&right);
    AffineConstraint constraint{CONSTRAINT_LT};
    //convention : treat left as primary affine and right as constant affine
    Affine left;
    Affine right;
    bool evaluate(const map<string,int> &eval_map) const;
    [[nodiscard]] string to_string(bool te=false) const;
    void replace(const string& source,const Affine& target);
    bool operator==(const Constraint& that) const;
    bool operator!=(const Constraint& that) const;
    static bool is_same_pattern(const Constraint& a,const Constraint& b,map<string,string>& m) ;

};
/*
 * we dont need to judge two presburger formula is semantic equivalence,so there is no need representing formula as
 * principal disjunctive normal form or principal conjunctive normal form
 * we only need to specify one presburger formula is true or false by substitute variable with constant
 * so choose data structure as expression tree
 * todo: implement Formula more general like principal conjunctive normal form or infer variable bound from formula
 * reference : ISL
 */
/*
 * if we use OpBase , we build expression tree by hand
 */
//class FormulaOpBase{
//public:
//    LogicSymbol symbol;
//    Constraint* left;
//    Constraint* right;
//    int num_parameter{1};
//    FormulaOpBase(LogicSymbol symbol1,Constraint* c);
//    FormulaOpBase(LogicSymbol symbol1,Constraint* c,Constraint* c1);
//};
//class FormulaNeg:public FormulaOpBase{
//    explicit FormulaNeg(Constraint* c);
//};
//class FormulaBinary:public FormulaOpBase{
//    FormulaBinary(LogicSymbol symbol1,Constraint* c,Constraint* c1);
//};
using mix = std::variant<Constraint,LogicSymbol>;
class Formula{
    /*
     * Tree Node
     * input formal : general all any rather than string
     */
public:
    bool autogen{true};
    LogicSymbol symbol{ENTITY_CONSTRAINT};
    LogicSymbol same_symbol{ENTITY_CONSTRAINT};
    Constraint entity;
    Formula* left{nullptr};
    Formula* right{nullptr};
    Formula(LogicSymbol symbol,Formula* left,Formula* right);
    explicit Formula(Constraint  c);
    // create tree : postfix expression
    //create tree: complete infix sequence,complete postfix sequence,complete medium sequence
    //create tree: infix sequence and medium sequence
    //create tree: other situation ......
    static Formula* create_general(const vector<mix>& postfix);
    static Formula* create_all(const vector<Constraint>& v);
    static Formula* create_any(const vector<Constraint>& v);
    bool evaluate(const map<string,int> &eval_map) const;
    Formula* conjunction(Formula* that);
    Formula* disjunction(Formula* that);
    [[nodiscard]] string to_string(bool te=false) const;// tensor expression if_else_then(expr,,)
    [[nodiscard]] string to_string_c_style(bool te=false) const ;
    void replace(const string& source,const Affine& target);
    bool operator==(const Formula& that) const;
    bool operator!=(const Formula& that) const;
    static bool is_same_pattern(const Formula& a,const Formula& b,map<string,string>& m);
private:
    //helper function
    static void do_binary(stack<Formula*>& s,LogicSymbol symbol1);

    [[nodiscard]] string to_same_string(bool te=false) const;
};

#endif //AFFINE3_FORMULA_H
