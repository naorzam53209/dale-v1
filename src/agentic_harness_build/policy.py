"""Policy expressions from the JSON Agents spec (Appendix B).

    tool.name == 'fetch_page' && args.url !~ '^https://'

An expression is parsed once with `compile_expression` and then evaluated against a
context dict such as {"tool": {...}, "args": {...}, "agent": {...}}.
"""

import operator
import re

ORDERING = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
KEYWORDS = {"and", "or", "not", "in", "contains", "starts_with", "ends_with", "true", "false", "null"}
SYMBOLS = ["==", "!=", ">=", "<=", "!~", "&&", "||", ">", "<", "~", "(", ")", "[", "]", ",", "."]
COMPARISONS = {"==", "!=", ">", "<", ">=", "<=", "~", "!~", "in", "not in", "contains", "starts_with", "ends_with"}


class PolicyError(ValueError):
    pass


# --- tokenizer -----------------------------------------------------------------


def _tokenize(text: str) -> list[tuple[str, object]]:
    tokens, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch == "'":
            value, i = [], i + 1
            while i < len(text) and text[i] != "'":
                if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == "'":
                    i += 1  # \' is an escaped quote; other backslashes stay (regexes need them)
                value.append(text[i])
                i += 1
            if i >= len(text):
                raise PolicyError(f"unterminated string in: {text}")
            tokens.append(("str", "".join(value)))
            i += 1
        elif ch.isdigit() or (ch == "-" and i + 1 < len(text) and text[i + 1].isdigit()):
            match = re.match(r"-?\d+(\.\d+)?", text[i:])
            tokens.append(("num", float(match[0]) if match[1] else int(match[0])))
            i += len(match[0])
        elif ch.isalpha() or ch == "_":
            match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[i:])
            word = match[0]
            tokens.append(("kw" if word in KEYWORDS else "id", word))
            i += len(word)
        else:
            symbol = next((s for s in SYMBOLS if text.startswith(s, i)), None)
            if symbol is None:
                raise PolicyError(f"unexpected character {ch!r} in: {text}")
            tokens.append(("sym", symbol))
            i += len(symbol)
    return tokens


# --- parser: builds a tree of small Python functions ----------------------------


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokenize(text)
        self.pos = 0

    def peek(self, offset: int = 0):
        index = self.pos + offset
        return self.tokens[index] if index < len(self.tokens) else (None, None)

    def accept(self, *values) -> bool:
        if self.peek()[1] in values and self.peek()[0] in ("sym", "kw"):
            self.pos += 1
            return True
        return False

    def expect(self, value):
        if not self.accept(value):
            raise PolicyError(f"expected {value!r} at token {self.pos} in: {self.text}")

    def parse(self):
        node = self.or_expr()
        if self.pos != len(self.tokens):
            raise PolicyError(f"unexpected {self.peek()[1]!r} in: {self.text}")
        return node

    def or_expr(self):
        node = self.and_expr()
        while self.accept("||", "or"):
            left, right = node, self.and_expr()
            node = lambda ctx, l=left, r=right: l(ctx) or r(ctx)
        return node

    def and_expr(self):
        node = self.unary()
        while self.accept("&&", "and"):
            left, right = node, self.unary()
            node = lambda ctx, l=left, r=right: l(ctx) and r(ctx)
        return node

    def unary(self):
        if self.accept("not"):
            inner = self.unary()
            return lambda ctx: not inner(ctx)
        return self.primary()

    def primary(self):
        if self.accept("("):
            node = self.or_expr()
            self.expect(")")
            return node
        kind, value = self.peek()
        if kind == "id":
            return self.comparison()
        literal = self.value()
        return lambda ctx: bool(literal)

    def comparison(self):
        path = self.accessor()
        if self.peek()[1] == "not" and self.peek(1)[1] == "in":
            self.pos += 2
            op = "not in"
        else:
            op = self.peek()[1]
            if op not in COMPARISONS or self.peek()[0] not in ("sym", "kw"):
                raise PolicyError(f"expected an operator after '{'.'.join(map(str, path))}' in: {self.text}")
            self.pos += 1
        expected = self.value()
        if op in ("~", "!~"):
            try:
                expected = re.compile(expected)
            except (re.error, TypeError) as e:
                raise PolicyError(f"bad regex {expected!r} in: {self.text} ({e})") from e
        return lambda ctx: _compare(_resolve(ctx, path), op, expected)

    def accessor(self) -> list:
        path = [self.peek()[1]]
        self.pos += 1
        while True:
            if self.accept("."):
                kind, name = self.peek()
                if kind not in ("id", "kw"):
                    raise PolicyError(f"expected a name after '.' in: {self.text}")
                path.append(name)
                self.pos += 1
            elif self.accept("["):
                kind, index = self.peek()
                if kind not in ("num", "str"):
                    raise PolicyError(f"expected an index inside [] in: {self.text}")
                path.append(index)
                self.pos += 1
                self.expect("]")
            else:
                return path

    def value(self):
        kind, value = self.peek()
        if kind in ("str", "num"):
            self.pos += 1
            return value
        if self.accept("true"):
            return True
        if self.accept("false"):
            return False
        if self.accept("null"):
            return None
        if self.accept("["):
            items = []
            if not self.accept("]"):
                items.append(self.value())
                while self.accept(","):
                    items.append(self.value())
                self.expect("]")
            return items
        raise PolicyError(f"expected a value at token {self.pos} in: {self.text}")


# --- evaluation ----------------------------------------------------------------


def _resolve(ctx, path: list):
    """Walk a.b[0].c through dicts and lists. Missing keys give None instead of failing."""
    value = ctx
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and -len(value) <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def _compare(actual, op: str, expected) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op in ("in", "not in"):
        found = isinstance(expected, list) and actual in expected
        return found if op == "in" else not found
    if op == "contains":
        if isinstance(actual, list):
            return expected in actual
        return isinstance(actual, str) and str(expected) in actual
    if op in ("~", "!~"):
        matched = isinstance(actual, str) and expected.search(actual) is not None
        return matched if op == "~" else not matched
    if op in ("starts_with", "ends_with"):
        if not isinstance(actual, str):
            return False
        return actual.startswith(expected) if op == "starts_with" else actual.endswith(expected)
    try:  # >, <, >=, <= on values that can't be ordered (e.g. None vs 5) just don't match
        return ORDERING[op](actual, expected)
    except TypeError:
        return False


def compile_expression(text: str):
    """Parse a policy expression and return a function: context dict -> bool."""
    return _Parser(text).parse()
