# -*- coding:utf-8 -*-
###
# File:  monads.py
# Created Date: 29/10/2025 07:47:17
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 29/05/2026 08:48:18
# Modified By: Lucas de Jesus 
# -----
# Copyright (c) 2026
# 
# This file is subject to the terms and conditions defined in
# the 'LICENSE.txt' file found in the root of this source tree.
# Please read LICENSE.txt for full copyright and licensing details.
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###


from __future__ import annotations
import jax
from dataclasses import dataclass
from typing import Self, Any, List, Callable, TypeVar, Generic, Tuple, TypeVarTuple, Unpack

# Definimos T (tipo atual) e U (tipo de destino após uma função)
T = TypeVar("T")
U = TypeVar("U")

D = TypeVar("D")
E = TypeVar("E")    #enviroment type

Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")


@dataclass(frozen=True)
class Reader(Generic[T, U]):
    operation: Callable[[T], U]

    @staticmethod
    def lift(operation: Callable[[D], E])->"Reader[D, E]":
        """ 
        operation:: rng ->  a
        """
        return Reader(operation)
    
    @staticmethod
    def pure(x: D)->"Reader[object, D]":
        return Reader(lambda _: x)
    

    def eval(self, data: T)->U:
        return self.operation(data)

    def map(self, f: Callable[[U], D])->"Reader[T, D]":
        """ 
        :: RBO rng a -> (b -> c) -> RBO rng c
        f:: b -> c
        """

        def func(data: T):
            a = self.eval(data)
            return f(a)
        
        return Reader(func)
    
    def bind(self, bind_fn: Callable[[U], "Reader[T, D]"])->"Reader[T, D]":
        """ 
        :: RBO rng a -> b -> RBO rng c -> RBO rng c
        bind_fn:: b -> PO r c
        """

        def func(data: T):
            a = self.eval(data)
            return bind_fn(a).eval(data)
        
        return Reader(func)


@dataclass(frozen=True)
class RngBoundOperation(Generic[T]):
    operation: Callable[[jax.Array], T]

    @staticmethod
    def lift(operation: Callable[[jax.Array], U])->"RngBoundOperation[U]":
        """ 
        operation:: rng ->  a
        """
        return RngBoundOperation(operation)
    
    @staticmethod
    def pure(x: U)->"RngBoundOperation[U]":
        return RngBoundOperation(lambda _: x)
    

    def eval(self, rng: jax.Array)->T:
        return self.operation(rng)

    def map(self, f: Callable[[T], U])->"RngBoundOperation[U]":
        """ 
        :: RBO rng a -> (b -> c) -> RBO rng c
        f:: b -> c
        """

        def func(rng: jax.Array):
            a = self.eval(rng)
            return f(a)
        
        return RngBoundOperation(func)
    
    def bind(self, bind_fn: Callable[[T], "RngBoundOperation[U]"])->"RngBoundOperation[U]":
        """ 
        :: RBO rng a -> b -> RBO rng c -> RBO rng c
        bind_fn:: b -> PO r c
        """

        def func(rng: jax.Array):
            rng1, rng2 = jax.random.split(rng) #cuida de splitar as chaves
            a = self.eval(rng1)
            return bind_fn(a).eval(rng2)
        
        return RngBoundOperation(func)

@dataclass(frozen=True)
class State(Generic[T, U]):
    computation: Callable[[T], Tuple[T, U]]

    """
    State s r = State(s -> (r, s))
    Associa uma uma computação que recebe um estado s, e retorna o resultado
    de tal computação r e um novo estado. Estamos associando necessariamente
    o resultado de uma computação com um estado.
    """

    @staticmethod
    def lift(computation: Callable[[T], Tuple[T, U]])->"State[T, U]":
        """
        computation:: s -> (s, r)
        """
        return State(computation)

    @staticmethod
    def pure(r:U)->"State[T, U]":
        return State(lambda s: (s, r))

    def map(self, f: Callable[[U], D])->State[T, D]:
        """
        map:: State s a -> (a -> b) -> State s b
        f:: a -> b
        a,b:: s -> (r, s)
        """

        def new_computation(state: T):
            # obtemos o resultado da computação (isto é o termo 'a')
            new_state, a = self.computation(state)

            # retornamos o resultado a aplicação de f, ou seja 'b', e o novo estado
            # (b, new_state)
            return new_state, f(a)

        # retornamos agora State s b, uma vez que new_computation:: s->(b,s)
        return State(new_computation)

    def bind(self, f:Callable[[U], "State[T, U]"])->State[T, U]:
        """
        bind:: State s a -> (a -> State s b) -> State s b
        f:: a -> State s b
        """

        def new_computation(state: T):
            # obtem o resultado da computação, 'a'
            new_state, a = self.computation(state)

            # obtemos State s b, por meio da aplicação de f em 'a'.
            # contudo new_computation:: s -> (b, s), portanto aplicamos a computação
            # contida em State s b por meio do run para obtermos (b, new_state)
            return f(a).run(new_state)

        # encapsulamos novamente para obter State s b final
        return State(new_computation)

    def run(self, state: T)->Tuple[T, U]:
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
class ReaderWriterM(Generic[E, T]):
    computation: Callable[[E, T], Tuple[Any, tuple]]
    
    @classmethod
    def pure(cls,  value)->Self:
        """
        computation:: e, i -> value, ()
        onde extra é uma tupla
        """
        return cls(lambda env, input: (value, ()))
    
    def map(self, func):
        """
        :: WriterMonad v e -> x -> y -> Writer Monad w e
        func:: x -> y
        """

        def new_computation(env, input):
            output, log = self.computation(env, input)
            return func(output), log
        
        return ReaderWriterM(new_computation)


    def bind(self, func):
        """
        :: WriterMonad v e -> x -> WriterMonad y, f -> WriterMonad w concat(e, f)
        func:: x -> WriterMonad y, f
        """

        def new_computation(env, input):
            output, log = self.computation(env, input)

            value2, log2 = func(output).run(env, input)

            return value2, (*log, *log2)
        
        return ReaderWriterM(new_computation)
    
    def run(self, env, input):
        return self.computation(env, input)