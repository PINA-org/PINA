import torch
from pina.problem import BaseProblem
from pina.condition import GraphInputTargetCondition, InputTargetCondition
from pina.problem.zoo import SupervisedProblem
from pina.graph import RadiusGraph


def test_constructor():
    input_ = torch.rand((100, 10))
    output_ = torch.rand((100, 10))
    problem = SupervisedProblem(input_=input_, output_=output_)
    assert isinstance(problem, BaseProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
    assert list(problem.conditions.keys()) == ["data"]
    assert isinstance(problem.conditions["data"], InputTargetCondition)


def test_constructor_graph():
    x = torch.rand((100, 10))
    pos = torch.rand((100, 2))
    input_ = RadiusGraph(x=x, pos=pos, radius=0.2, edge_attr=True)
    output_ = torch.rand((100, 10))
    problem = SupervisedProblem(input_=input_, output_=output_)
    assert isinstance(problem, BaseProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
    assert list(problem.conditions.keys()) == ["data"]
    assert isinstance(problem.conditions["data"], GraphInputTargetCondition)
    assert problem.conditions["data"].graph is input_
    assert problem.conditions["data"].output_key == "y"
    assert torch.equal(input_.y, output_)
