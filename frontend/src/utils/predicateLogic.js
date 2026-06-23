
/* 
    Utility for boolean predicate logic comparison.
    Used to detect duplicate rules with logically equivalent predicates (e.g., "A AND B" == "B AND A").
*/

export const tokenize = (text) => (text || '').match(/\(|\)|AND|OR|NOT|[A-Za-z0-9_]+/g) || [];

const getPrecedence = (op) => {
    switch (op) {
        case 'NOT': return 3;
        case 'AND': return 2;
        case 'OR': return 1;
        default: return 0;
    }
};

const isOperator = (token) => ['AND', 'OR', 'NOT'].includes(token);

/**
 * Converts infix tokens to Reverse Polish Notation (RPN) using Shunting-yard algorithm.
 */
const toRPN = (tokens) => {
    const outputQueue = [];
    const operatorStack = [];

    for (const token of tokens) {
        if (token === '(') {
            operatorStack.push(token);
        } else if (token === ')') {
            while (operatorStack.length && operatorStack[operatorStack.length - 1] !== '(') {
                outputQueue.push(operatorStack.pop());
            }
            operatorStack.pop(); // Pop '('
        } else if (isOperator(token)) {
            while (
                operatorStack.length &&
                operatorStack[operatorStack.length - 1] !== '(' &&
                (
                    (token !== 'NOT' && getPrecedence(operatorStack[operatorStack.length - 1]) >= getPrecedence(token)) ||
                    (token === 'NOT' && getPrecedence(operatorStack[operatorStack.length - 1]) > getPrecedence(token))
                )
            ) {
                outputQueue.push(operatorStack.pop());
            }
            operatorStack.push(token);
        } else {
            // Identifier / Operand
            outputQueue.push(token);
        }
    }
    while (operatorStack.length) {
        outputQueue.push(operatorStack.pop());
    }
    return outputQueue;
};

/**
 * Evaluates RPN with a given boolean context.
 */
const evaluateRPN = (rpn, context) => {
    const stack = [];
    for (const token of rpn) {
        if (token === 'NOT') {
            const a = stack.pop();
            stack.push(!a);
        } else if (token === 'AND') {
            const b = stack.pop();
            const a = stack.pop();
            stack.push(a && b);
        } else if (token === 'OR') {
            const b = stack.pop();
            const a = stack.pop();
            stack.push(a || b);
        } else {
            // Variable
            stack.push(!!context[token]);
        }
    }
    return stack.length ? stack[0] : false;
};

const getVariables = (tokens) => {
    const vars = new Set();
    tokens.forEach(t => {
        if (!isOperator(t) && t !== '(' && t !== ')') {
            vars.add(t);
        }
    });
    return Array.from(vars).sort();
};

/**
 * Checks if two predicate strings are methodologically equivalent.
 * Uses Truth Table comparison.
 * @param {string} pred1Str - First predicate
 * @param {string} pred2Str - Second predicate
 * @returns {boolean} - True if logically equivalent
 */
export const arePredicatesEquivalent = (pred1Str, pred2Str) => {
    if (pred1Str === pred2Str) return true;
    
    // 1. Tokenize & RPN
    const tokens1 = tokenize(pred1Str);
    const tokens2 = tokenize(pred2Str);

    // If parsing fails or empty, simple compare
    if (!tokens1.length || !tokens2.length) return pred1Str === pred2Str;

    const rpn1 = toRPN(tokens1);
    const rpn2 = toRPN(tokens2);

    // 2. Identify Variables
    const vars1 = getVariables(tokens1);
    const vars2 = getVariables(tokens2);

    // The set of variables must be identical for our purposes (strict duplicate check)
    // If logic differs by variable, it's different.
    if (vars1.length !== vars2.length) return false;
    for (let i = 0; i < vars1.length; i++) {
        if (vars1[i] !== vars2[i]) return false;
    }
    
    const variables = vars1;

    // Safety Cap: If > 12 variables, fallback to strict string equality to avoid performance hang
    if (variables.length > 12) {
        return pred1Str === pred2Str;
    }

    // 3. Truth Table comparison
    const limit = 1 << variables.length;
    for (let i = 0; i < limit; i++) {
        const context = {};
        for (let j = 0; j < variables.length; j++) {
            context[variables[j]] = (i >> j) & 1 ? true : false;
        }

        const res1 = evaluateRPN(rpn1, context);
        const res2 = evaluateRPN(rpn2, context);

        if (res1 !== res2) {
            return false;
        }
    }

    return true;
};
