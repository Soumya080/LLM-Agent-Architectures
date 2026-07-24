class DecisionPolicy :
    def __init__(self, GoalNode, GoalTree, PlannerConsraints, PlanningContext, Memory, PlanningState):
        self.goalnode = GoalNode
        self.goaltree = GoalTree
        self.planner_constraints = PlannerConsraints
        self.planning_context = PlanningContext
        self.memory = Memory
        self.planning_state = PlanningState
    
    def should_decompose(self, goal_node=None, goal_tree=None, planning_context=None, memory=None, planning_state=None) -> dict:
        """Main API: Determine if goal should be decomposed"""
        if goal_node is not None:
            self.goalnode = goal_node
        if goal_tree is not None:
            self.goaltree = goal_tree
        if planning_context is not None:
            self.planning_context = planning_context
        if memory is not None:
            self.memory = memory
        if planning_state is not None:
            self.planning_state = planning_state
        try:
            constraints = self._evaluate_constraints()
            atomicity = self._evaluate_atomicity()
            budget = self._evaluate_budget()
            complexity = self._estimate_complexity()
            scores = self._aggregate_scores(constraints, atomicity, budget, complexity)
            decision = self._make_decision(scores)
            
            result = {
                "decision": decision,
                "decompose": decision == "YES",
                "reasoning": "Based on constraint, atomicity, budget, and complexity evaluation",
                "scores": scores
            }
        except ValueError:
            result = {
                "decision": "NO",
                "decompose": False,
                "reasoning": "Constraint violation detected"
            }
        
        return result
    
    def _evaluate_constraints(self, violations=None) -> str:
        """Check constraint violations"""
        if violations is None:
            state = {
                "depth": self.goalnode.depth if self.goalnode else 0,
                "nodes": len(self.goaltree.nodes) if self.goaltree else 0,
            }
            if isinstance(self.planning_state, dict):
                state.update(self.planning_state)
            if self.planner_constraints:
                violations = self.planner_constraints.validate(state).get("violations", [])
            else:
                violations = []
        
        if "Maximum tree depth exceeded." in violations:
            raise ValueError("Depth Exceeded")
        
        elif "Token budget exceeded" in violations:
            raise ValueError("Budget Exceeded")
        
        elif "Maximum node count exceeded." in violations:
            raise ValueError("Maximum Node reached")
        
        else:
            return "proceed"
    
    def _evaluate_atomicity(self) -> bool:
        """Check if all tasks are already atomic"""
        if self.goalnode and self.goalnode.status == "atomic":
            return False
        return True
    
    def _evaluate_budget(self, violations=None) -> str:
        """Check budget constraints"""
        if violations is None:
            state = {
                "depth": self.goalnode.depth if self.goalnode else 0,
                "nodes": len(self.goaltree.nodes) if self.goaltree else 0,
            }
            if isinstance(self.planning_state, dict):
                state.update(self.planning_state)
            if self.planner_constraints:
                violations = self.planner_constraints.validate(state).get("violations", [])
            else:
                violations = []
        
        if "Token budget exceeded" in violations:
            raise ValueError("Budget Exceeded")
        
        elif "Planning time budget exceeded." in violations:
            raise ValueError("Time Exceed")
        
        else:
            return "proceed"
    
    def _estimate_complexity(self):
        """Estimate goal complexity"""
        return 0.5
    
    def _aggregate_scores(self, constraints, atomicity, budget, complexity) -> dict:
        """Aggregate evaluation scores"""
        return {
            "constraints": 1.0 if constraints == "proceed" else 0.0,
            "atomicity": 1.0 if atomicity else 0.0,
            "budget": 1.0 if budget == "proceed" else 0.0,
            "complexity": complexity
        }
    
    def _make_decision(self, scores) -> str:
        """
        Decision Flow:
        ↓ Check Constraints
        ↓ Check Atomicity
        ↓ Check Budget
        ↓ Estimate Complexity
        ↓ Aggregate Scores
        ↓ Make Decision
        """
        if (scores["constraints"] == 1.0 and 
            scores["atomicity"] == 1.0 and 
            scores["budget"] == 1.0):
            decision = "YES"
        else:
            decision = "NO"
        
        return decision
    
    def __repr__(self):
        return f"DecisionPolicy(goal_node={self.goalnode}, goal_tree={self.goaltree})"
