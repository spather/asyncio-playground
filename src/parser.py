"""
Implements a parser for the following BNF grammar:

<expression> ::= <numeric_literal> | <expression> <operator> <expression>
<operator> ::= + | - | * | /
<numeric_literal> ::= [0-9]+
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import List, Optional, Type, TypeVar, Union


# Token types
@dataclass
class Number:
    value: int


@dataclass
class Operator:
    value: str


Token = Number | Operator


# AST Nodes
@dataclass
class NumericLiteral:
    value: int


@dataclass
class BinaryExpression:
    left: Union[NumericLiteral, BinaryExpression]
    right: Union[NumericLiteral, BinaryExpression]
    operator: str


ASTNode = NumericLiteral | BinaryExpression


@dataclass
class Tokenizer:
    def __init__(self, s: str):
        self.s = s
        self.cursor = 0

    def hasMoreTokens(self) -> bool:
        return self.cursor < len(self.s)

    def getNextToken(self) -> Optional[Token]:
        if not self.hasMoreTokens():
            return None

        # Whitespace
        m = re.match("^\s+", self.s[self.cursor :])
        if m is not None:
            # Just move past whitespace
            self.cursor += m.end()

        # Digits
        m = re.match("\d+", self.s[self.cursor :])
        token: Optional[Token] = None

        if m is not None:
            token = Number(int(m.group(0)))
            self.cursor += m.end()
            return token

        # Operators
        m = re.match("[\+\-\*\/]", self.s[self.cursor :])
        if m is not None:
            token = Operator(m.group(0))
            self.cursor += m.end()
            return token

        raise SyntaxError(
            f"Could not extract any tokens starting at position {self.cursor}: {self.s[self.cursor:self.cursor+10]}"
        )


TToken = TypeVar("TToken", bound=Token)


class Parser:
    def __init__(self):
        self.tokenizer = Tokenizer("")
        self.lookahead: Optional[Token] = None

    def parse(self, s: str):
        self.tokenizer = Tokenizer(s)
        self.lookahead = self.tokenizer.getNextToken()

        return self.expression()

    def expression(self):
        left: Union[NumericLiteral, BinaryExpression] = self.numeric_literal()

        while isinstance(self.lookahead, Operator):
            op = self.eat(Operator)
            right = self.numeric_literal()
            left = BinaryExpression(left, right, op.value)

        return left

    def numeric_literal(self) -> NumericLiteral:
        token = self.eat(Number)
        return NumericLiteral(token.value)

    def eat(self, token_type: Type[TToken]) -> TToken:
        token = self.lookahead

        if token is None:
            raise SyntaxError(f"Unexpected end of input, expected {token_type}")

        if not isinstance(token, token_type):
            raise SyntaxError(f"Unexpected token {type(token)}, expected {token_type}")

        self.lookahead = self.tokenizer.getNextToken()
        return token


# Because this is for interview practice, write the eval function
# both recursively and iteratively.


def eval_ast_recursive(ast: ASTNode) -> int:
    if isinstance(ast, NumericLiteral):
        return ast.value

    if isinstance(ast, BinaryExpression):
        if ast.operator == "+":
            return eval_ast_recursive(ast.left) + eval_ast_recursive(ast.right)
        elif ast.operator == "-":
            return eval_ast_recursive(ast.left) - eval_ast_recursive(ast.right)
        elif ast.operator == "*":
            return eval_ast_recursive(ast.left) * eval_ast_recursive(ast.right)
        elif ast.operator == "/":
            return eval_ast_recursive(ast.left) // eval_ast_recursive(ast.right)

    raise ValueError(f"Unknown AST type {type(ast)}")


def eval_ast_iterative(ast: ASTNode) -> int:
    stack: List[Union[int, str, ASTNode]] = [ast]
    results: List[int] = []

    while len(stack) > 0:
        node = stack.pop()

        if isinstance(node, NumericLiteral):
            results.append(node.value)
        elif isinstance(node, BinaryExpression):
            stack.extend([node.operator, node.left, node.right])
        elif isinstance(node, str):
            right_operand = results.pop()
            left_operand = results.pop()

            if node == "+":
                results.append(right_operand + left_operand)
            elif node == "-":
                results.append(right_operand - left_operand)
            elif node == "*":
                results.append(right_operand * left_operand)
            elif node == "/":
                results.append(right_operand // left_operand)
            else:
                raise ValueError(f"Unknown operator on stack {node}")

    return results.pop()


async def main():
    test_str = "123"
    p = Parser()
    ast = p.parse(test_str)
    print(test_str)
    print(ast)
    print(f" == {eval_ast_recursive(ast)}")
    print()

    assert ast == NumericLiteral(123)
    assert eval_ast_recursive(ast) == 123
    assert eval_ast_iterative(ast) == eval_ast_recursive(ast)

    test_str = "123 + 456"
    ast = p.parse(test_str)
    print(test_str)
    print(ast)
    print(f" == {eval_ast_recursive(ast)}")
    print()

    assert ast == BinaryExpression(NumericLiteral(123), NumericLiteral(456), "+")
    assert eval_ast_recursive(ast) == 123 + 456
    assert eval_ast_iterative(ast) == eval_ast_recursive(ast)

    test_str = "123 + 456 + 789"
    ast = p.parse(test_str)
    print(test_str)
    print(ast)
    print(f" == {eval_ast_recursive(ast)}")
    print()

    assert ast == BinaryExpression(
        BinaryExpression(NumericLiteral(123), NumericLiteral(456), "+"),
        NumericLiteral(789),
        "+",
    )
    assert eval_ast_recursive(ast) == 123 + 456 + 789
    assert eval_ast_iterative(ast) == eval_ast_recursive(ast)


if __name__ == "__main__":
    asyncio.run(main())
