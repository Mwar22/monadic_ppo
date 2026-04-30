from __future__ import annotations
from dataclasses import dataclass
from typing import Self, Any, List, Callable, TypeVar, Generic, Tuple, TypeVarTuple, Unpack

# Definimos T (tipo atual) e U (tipo de destino após uma função)
T = TypeVar("T")
U = TypeVar("U")

Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")
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
  
@dataclass(frozen=True)
class MaybeM(Generic[T]):
    value: T | None

    @classmethod
    def nothing(cls):
        return cls(None)
    
    @classmethod
    def just(cls, value: T)-> MaybeM[T]:
        return cls(value)
    
    def is_nothing(self):
        return self.value is None
    
    def map(self, func:Callable[[T], U])-> MaybeM[U]:
        """
        :: Maybe a -> (x -> y) -> Maybe b
        func::  value -> value
        """
        if self.value is None:
            return MaybeM.nothing()
        
        b = func(self.value)
        if b is  None:
            return MaybeM.nothing()
        
        return MaybeM.just(b)
    

    def bind(self, func: Callable[[T], MaybeM[U]])->MaybeM[U]:
        """
        :: Maybe a -> (x -> Maybe y) - > Maybe b
        """
        if self.value is None:
            return MaybeM.nothing()

        return func(self.value)
    
    def unzip(self: MaybeM[ListM[U]]):
        """
        Transforma uma Monad de Tupla em uma Tupla de Monads.
        Ex: Maybe[(List[int], List[str])] -> (Maybe[List[int]], Maybe[List[str]])
        """
        if self.value is None:
            return ListM.pure([MaybeM[U].nothing()])
        
        eval_value = lambda v: MaybeM[U].nothing() if v is None else MaybeM[U].just(v)
        return ListM.pure([eval_value(v) for v in self.value.data])
    
    def __repr__(self) -> str:
        return f"Just({self.value})" if not self.is_nothing() else "Nothing"


@dataclass(frozen=True)
class ListM(Generic[T]):
    data: List[T]

    @classmethod
    def pure(cls, *args:T)->ListM[T]:
        return cls(list(args))
    
    def map(self, func: Callable[[T], U])->ListM[U]:
        """
        :: ListMonad l -> i -> j -> ListMonad m
        func:: i->j
        """
        return ListM([func(i) for i in self.data])
    
    def bind(self, func: Callable[[T], ListM[U]])->ListM[U]:
        """
        :: ListMonad  a -> (x -> ListMonad  y) - > ListMonad j
        """

        new_values = []
        for v in self.data:
            new_values.extend(func(v).data)

        return ListM.pure(new_values)

    
    def imap(self, func: Callable[[int, T], U])->ListM[U]:
        """ indexed map.

        same as map, but func also receives the index in the list
        :: ListMonad l -> (idx, x) -> y -> ListMonad m
        func:: (idx, x->y
        """
        return ListM([func(i, v) for i, v in enumerate(self.data)])
    
    def ibind(self, func: Callable[[int, T], ListM[U]])->ListM[U]:
        """ Indexed bind.
        :: ListMonad  a -> (idx, x) -> ListMonad  y) - > ListMonad j
        """

        new_values = []
        for i, v in enumerate(self.data):
            new_values.extend(func(i, v).data)

        return ListM.pure(new_values)
    
    def __repr__(self) -> str:
        return f"ListMonad({self.data})"
    

@dataclass(frozen=True)
class WriterM:
    computation: Any
    
    @classmethod
    def pure(cls,  computation)->Self:
        """
        computation:: x -> y, extra
        onde extra é uma tupla
        """
        return cls(computation)
    
    def map(self, func):
        """
        :: WriterMonad v e -> x -> y -> Writer Monad w e
        func:: x -> y
        """

        def new_computation(input):
            output, extra = self.computation(input)
            return func(output), extra
        
        return WriterM(new_computation)


    def bind(self, func):
        """
        :: WriterMonad v e -> x -> WriterMonad y, f -> Writer Monad w concat(e, f)
        func:: x -> WriterMonad y, f
        """

        def new_computation(input):
            output, extra = self.computation(input)
            output2, extra2 = func(output)

            #extra o conteudo extra original e novos e concatena novamente em uma tupla
            return output2, (*extra, *extra2)
        
        return WriterM(new_computation)
    
    def run(self, input):
        return self.computation(input)