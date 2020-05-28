from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import argparse

parser = argparse.ArgumentParser(description='Provision AML Compute Resources.')

parser.add_argument('-n', '--name', dest='name', 
                    help='Compute resource name', required=True)
parser.add_argument('-w', '--workspace', dest='workspace', 
                    help='Workspace name', required=True)
parser.add_argument('-r', '--resource-group', dest='resource_group', 
                    help='Resource group name', required=True)
parser.add_argument('-s', '--suscription-id', dest='subscription_id', 
                    help='Subscription Id', required=True)

args = parser.parse_args()

# Get a reference to our workspace
ws = Workspace.get(name=args.workspace,
					subscription_id=args.subscription_id,
					resource_group=args.resource_group)

# Verify that cluster does not exist already
try:
    cpu_cluster = AmlCompute(workspace=ws, name=args.name)
    print('Cluster already exists.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_D3_v2', max_nodes=2)
    cpu_cluster = AmlCompute.create(ws, args.name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)