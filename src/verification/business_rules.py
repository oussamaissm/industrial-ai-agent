import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.simulator.factory import FactorySimulator
from src.verification.schemas import ScheduleProposal

def verify_schedule(proposal: ScheduleProposal, simulator: FactorySimulator) -> dict:
    machine = simulator.get_machine_status(proposal.machine_id)
    
    if isinstance(machine, dict) and machine.get("error"):
        return {"valid": False, "reason": "Invalid machine ID"}
        
    if machine["status"] == "down":
        return {"valid": False, "reason": "Machine is currently down"}
        
    if proposal.estimated_hours > machine["capacity_hours"]:
        return {"valid": False, "reason": "Exceeds machine daily capacity"}
        
    return {"valid": True, "reason": "Passed all constraints"}
