# HUST quarantine opcode inspector

This directory is a deliberately narrow Docker build context. It inspects one
approved HUST pickle member with Python's `pickletools` parser and never calls
`pickle.load` or `pickle.loads`.

It is **not** the HUST converter. The output contains only pickle protocol,
opcode counts and referenced global names. It does not establish field names,
units, cell identity, SOH, EOL or RUL.

Runtime prerequisites:

1. an explicitly approved frozen inventory;
2. a second archive audit with `ready_for_conversion=true`;
3. a read-only, workspace-external input directory;
4. a read-only job directory containing `job.json`;
5. a new empty output directory;
6. Docker with runtime networking disabled.

The base image currently uses an exact Python patch tag but not a registry
digest because this workstation is offline. Before any real run, resolve and
record the image digest in the Dockerfile and review it as a separate change.
