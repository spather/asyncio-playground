"""
Note: I had wanted to use this example to show how generators and co-routines
      could simplify stateful code otherwise implemented in classes. I wrote
      a simple parser for simple numeric expressions in the standard way and
      then was going to convert it. Got as far as converting the tokenizer to
      a generator and then ran out of time / inspiration.

Implements a parser for the following BNF grammar:

<expression> ::= <numeric_literal> | <expression> <operator> <expression>
<operator> ::= + | - | * | /
<numeric_literal> ::= [0-9]+
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Iterator, List, Optional, Type, TypeVar, Union


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


def tokenizer_gen(s: str) -> Iterator[Optional[Token]]:
    cursor = 0

    while cursor < len(s):
        # Whitespace
        m = re.match("^\s+", s[cursor:])
        if m is not None:
            # Just move past whitespace
            cursor += m.end()

        # Digits
        m = re.match("\d+", s[cursor:])
        token: Optional[Token] = None

        if m is not None:
            token = Number(int(m.group(0)))
            cursor += m.end()
            yield token
            continue

        # Operators
        m = re.match("[\+\-\*\/]", s[cursor:])
        if m is not None:
            token = Operator(m.group(0))
            cursor += m.end()
            yield token
            continue

        raise SyntaxError(
            f"Could not extract any tokens starting at position {cursor}: {s[cursor:cursor+10]}"
        )

    # No more tokens
    yield None


TToken = TypeVar("TToken", bound=Token)


class Parser:
    def __init__(self):
        self.tokenizer = tokenizer_gen("")
        self.lookahead: Optional[Token] = None

    def parse(self, s: str):
        self.tokenizer = tokenizer_gen(s)
        self.lookahead = next(self.tokenizer)

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

        self.lookahead = next(self.tokenizer)
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
