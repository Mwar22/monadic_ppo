class State:
    """
    State s r = State(s -> (r, s))
    Associa uma uma computação que recebe um estado s, e retorna o resultado
    de tal computação r e um novo estado. Estamos associando necessariamente
    o resultado de uma computação com um estado.
    """

    def __init__(self, computation):
        """
        computation:: s -> (r, s)
        """
        self.computation = computation

    def map(self, f):
        """
        map:: State s a -> (a -> b) -> State s b
        f:: a -> b
        a,b:: s -> (r, s)
        """

        def new_computation(state):
            # obtemos o resultado da computação (isto é o termo 'a')
            a, new_state = self.computation(state)

            # retornamos o resultado a aplicação de f, ou seja 'b', e o novo estado
            # (b, new_state)
            return f(a), new_state

        # retornamos agora State s b, uma vez que new_computation:: s->(b,s)
        return State(new_computation)

    def bind(self, f):
        """
        bind:: State s a -> (a -> State s b) -> State s b
        f:: a -> State s b
        """

        def new_computation(state):
            # obtem o resultado da computação, 'a'
            a, new_state = self.computation(state)

            # obtemos State s b, por meio da aplicação de f em 'a'.
            # contudo new_computation:: s -> (b, s), portanto aplicamos a computação
            # contida em State s b por meio do run para obtermos (b, new_state)
            return f(a).run(new_state)

        # encapsulamos novamente para obter State s b final
        return State(new_computation)

    def run(self, state):
        """
        run:: s -> (r, s)
        """
        return self.computation(state)
