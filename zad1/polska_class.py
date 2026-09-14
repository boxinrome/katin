import re

class Polish:
    def __init__(self, string: str, type_of_operation: str):
        self.string = string.strip()
        self.type = type_of_operation.strip()

    def polish(self) -> str:

        tokens = re.findall(r'\d+(?:\.\d+)?|[+\-*/^()]', self.string)
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        output = []
        stack = []

        for token in tokens:
            if token.replace('.', '', 1).isdigit():
                output.append(token)
            elif token in precedence:
                while (stack and stack[-1] != '(' and 
                       (precedence.get(stack[-1], 0) > precedence[token] or
                        (precedence.get(stack[-1], 0) == precedence[token] and token != '^'))):
                    output.append(stack.pop())
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack and stack[-1] == '(':
                    stack.pop()

        while stack:
            output.append(stack.pop())

        return " ".join(output)

    def common(self) -> str:
        stack = []
        priority = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
        tokens = self.string.split()

        for token in tokens:
            if token in ('(', ')'):
                continue
            if token not in priority:
                stack.append((token, 4))
            else:
                if len(stack) < 2:
                    raise ValueError("Некорректное выражение ОПЗ")
                right_exp, right_prio = stack.pop()
                left_exp, left_prio = stack.pop()

                current_prio = priority[token]

                if left_prio < current_prio:
                    left_exp = f"({left_exp})"

                if right_prio <= current_prio:
                    right_exp = f"({right_exp})"

                new_expr = f"{left_exp} {token} {right_exp}"
                stack.append((new_expr, current_prio))

        return stack[0][0] if stack else ""

    def evaluate_rpn(self) -> str:
        stack = []
        tokens = self.string.split()

        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '^': lambda a, b: a ** b,
        }

        for token in tokens:
            if token in ('(', ')'):
                continue

            if token in ops:
                if len(stack) < 2:
                    raise ValueError(f"Недостаточно операндов для операции '{token}'")
                b = stack.pop()
                a = stack.pop()
                if token == '/' and b == 0:
                    raise ZeroDivisionError("Деление на ноль")
                res = ops[token](a, b)
                stack.append(res)
            else:
                try:
                    stack.append(float(token))
                except ValueError:
                    raise ValueError(f"Неизвестный символ: '{token}'")

        if len(stack) != 1:
            raise ValueError("Некорректное выражение: в стеке осталось более одного значения")

        final_result = stack[0]
        return str(int(final_result) if final_result.is_integer() else final_result)

    def action(self):
        match self.type:
            case "to_polish" | "to polish":
                return self.polish()
            case "to_common" | "to common":
                return self.common()
            case "count":
                return self.evaluate_rpn()
            case _:
                return f"Неизвестный тип операции: {self.type}"