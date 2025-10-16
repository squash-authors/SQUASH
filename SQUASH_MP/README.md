# SQUASH_MP
This is the "multiprocessing" version of the SQUASH run-time (i.e. query-time) system. It can be executed on cloud-provisioned servers such as those provided by AWS EC2, local (on-premise) servers (physical or virtual) and even workstations/PCs, assuming they have sufficient processing capacity.

## Folder Structure
```script
- datasets: Used to store datasets and generated indices. One subfolder per dataset.
- logs: Runtime logs are automatically written here, and we recommend that command line output is also redirected here.
- output: General output area
- scripts: 
    - generate_querydata.py : Generates query sets, ground truth files and deployment files for a dataset      
- src:
    - coordinator.py: Coordinator class 
    - datamanager.py: DataManager class, provides support functionality for QueryAllocator/QueryProcessor classes
    - queryallocator.py: QueryAllocator class
    - queryprocessor.py: QueryProcessor class
    - treelauncher.py: TreeLauncher class, used by Coordinator and QueryProcessor to manage Lambda tree
- Driver script (free-standing):
    - smp_runner: This script enables configuration of runtime parameters and launch of a query session
```

## Dependencies
The following package versions were used for SQUASH_MP.
```script
- python==3.13.2
- boto3==1.37.10
- numpy==2.2.4
- bitarray==3.3.0
- faiss-cpu==1.11.0
```

## Instructions
Before running this code, SQUASH_BUILD must have been used to build the required SQUASH indexes and support files. The relevant folder should then be copied from SQUASH_BUILD/datasets into SQUASH_MP/datasets.  

If query data has not already been generated, the generate_querydata.py script can now be used. This builds the global query data file, creates an "allocators" sub-directory under the dataset, and populates it with *num_allocators* sub-directories containing the subset query data for each allocator. Note that using the (R)ebuild mode, the number of allocators can be increased/decreased and the query workload re-subsetted accordingly.

To execute SQUASH_MP, update the smp_runner.py script to specify:

**Dataset details** including path, name, number of vectors etc
**Runtime parameters** as described in SQLAYER    

Then run smp_runner.

