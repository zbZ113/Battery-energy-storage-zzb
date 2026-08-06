# HUST quarantine opcode inspector

This directory is a deliberately narrow Docker build context. It inspects one
approved HUST pickle member with Python's `pickletools` parser and never calls
`pickle.load` or `pickle.loads`.

It is **not yet** an approved HUST converter. The output contains only pickle protocol,
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

`convert.py` is a fail-closed semantic gate. With the current
`hust_mendeley_v2_layout_v1.json` it returns `BLOCKED_REVIEW` and writes
nothing. It deliberately contains no pickle deserialization code. A future
converter may only be implemented after every field, unit, stable cell identity,
and conversion rule in the layout has received explicit human approval.
