# Project Guidelines

## Scope and Source of Truth
- Use this as the workspace-wide instruction file.
- Link to existing docs instead of duplicating setup details:
  - `README.md` for Docker, deployment, tmux workflows, and demo procedures.
  - `wsl_readme.md` for WSL + USB passthrough notes.

## Architecture
- `xbee_dec_gnn/scripts/graph_server.py` is the central graph server process.
- `xbee_dec_gnn/xbee_dec_gnn/node.py` is the per-device decentralized GNN runtime.
- `xbee_dec_gnn/xbee_dec_gnn/utils/zigbee_comm.py` defines the Zigbee transport, message framing, handshake protocol, and publisher utilities.
- `xbee_dec_gnn/xbee_dec_gnn/decentralized_gnns/` contains decentralized message passing and pooling logic.
- `xbee_dec_gnn/xbee_dec_gnn/encoder.py` contains tensor serialization helpers used by node message exchange.

## Build and Run
- Build container image from repo root:
  - `docker build -t xbee_gnn_img .`
- Start container:
  - `./docker/run_docker.sh`
- Run central graph server (inside container):
  - `cd ~/other_ws/xbee_dec_gnn/xbee_dec_gnn/scripts`
  - `python3 graph_server.py --port=/dev/ttyUSB0 --baud=9600 --mode=load --gui`
- Run one node (inside container):
  - `cd ~/other_ws/xbee_dec_gnn/xbee_dec_gnn/xbee_dec_gnn`
  - `python3 node.py --port=/dev/ttyUSB0 --baud=9600 --model-path=/root/resources/models/MIDS_model.pth --num-nodes=5 --node-id=0`
- Multi-pane orchestration:
  - `cd ~/other_ws/xbee_dec_gnn/xbee_dec_gnn/launch`
  - `tmuxinator start -p tmux_launch_mids.yml local`

## Lint and Style
- Python target is 3.8; style and lint config lives in `xbee_dec_gnn/pyproject.toml`.
- Ruff conventions: 4-space indentation, max line length 120, double-quote formatting.
- Keep changes minimal and local; do not reformat unrelated files.

## Project Conventions
- Node naming is strict: `node_<id>` (for example, `node_0`). Address maps and neighbor resolution depend on this format.
- Graph server sends only per-node neighbors and per-node feature vector via GRAPH messages.
- Node waits for handshake completion before creating publishers.
- Message passing and pooling loops block until all active neighbors provide data, with 30-second timeouts.

## Zigbee Communication Interface (Canonical Summary)

### Roles
- Central:
  - Class: `ZigbeeCentralInterface`
  - Owns global `addr_map` (node name -> 64-bit XBee address).
  - Initiates handshake and sends GRAPH messages.
- Node:
  - Class: `ZigbeeNodeInterface`
  - Responds to discovery, receives `addr_map`, confirms, then exchanges MP/POOLING messages with neighbors.

### Topics and Message Types
- Topic enum (`Topic`):
  - `HANDSHAKE = 1`
  - `GRAPH = 2`
  - `MP = 3`
  - `POOLING = 4`
- Handshake steps (`HandshakeStep`):
  - `DISCOVERY = 1`, `REGISTER = 2`, `ACK = 3`, `CONFIRM = 4`
- Message classes:
  - `HandshakeMessage(step, sender_name, sender_addr, addr_map)`
  - `GraphMessage(neighbors, features)`
  - `DataExchangeMessage(sender_name, layer, round_id, iteration, data, shape, sources)`

### Payload and Wire Format
- Standard framed message path uses:
  - `encode_message(topic, msg) = [1 byte topic] + pickle.dumps(msg.to_payload(), protocol=5)`
- Decode path expects the same framing:
  - `decode_message(data)` reads `data[0]` as topic enum value and unpickles `data[1:]`.
- Tensor payloads in MP/POOLING use:
  - `pack_tensor()` / `unpack_tensor()` from `encoder.py` (float32 little-endian bytes + explicit shape).

### Addressing
- Broadcast constants in central interface:
  - 64-bit broadcast: `000000000000FFFF`
  - 16-bit broadcast: `FFFE`
- Unicast sends use `send_data_64_16(target_addr, UNKNOWN_ADDRESS, data)`.
- Central stores addresses as strings in `addr_map`; node names are expected to match `node_<id>`.

### Handshake Procedure
1. Central starts and repeatedly sends `DISCOVERY` every 5 seconds until all expected nodes register.
2. Each node on `DISCOVERY`:
   - Stores central address from the message.
   - Sends `REGISTER(sender_name=node_<id>, sender_addr=<my_64bit_addr>)`.
3. Central collects all `REGISTER` messages, builds `addr_map`, and unicasts `ACK(addr_map=...)` to each node.
4. Node on `ACK`:
   - Saves `addr_map`.
   - Sends `CONFIRM(sender_name=node_<id>)`.
   - Sets handshake-complete event.
5. Central waits for all `CONFIRM` messages and marks handshake complete.

### Runtime Communication Flow
1. Central loads/generates graph.
2. Central sends `GRAPH` to each node:
   - `neighbors`: list of node names (for example, `node_1`).
   - `features`: per-node feature vector (list of float).
3. Node receives GRAPH, updates local subgraph and `starting_data`.
4. For each GNN layer:
   - Node sends `MP` (`round_id`, `layer`, packed tensor data/shape) to active neighbors.
   - Node waits until all active neighbors provide data for that `(round_id, layer)` key.
5. For each pooling iteration:
   - Node sends `POOLING` (`iteration`, packed data/shape, optional `sources` for flooding mode).
   - Node waits until all active neighbors provide that iteration.
6. Node runs prediction and outputs MIDS decision.

### Reliability, Size Limits, and Timeouts
- Publish retries in `XBeePublisher.publish()`:
  - up to 5 attempts with exponential backoff (`0.05, 0.1, 0.2, 0.4, 0.8` seconds)
  - catches `TransmitException` and `TimeoutException`.
- Sync operations timeout defaults to 4.0 seconds (`set_sync_ops_timeout`).
- Handshake wait on node side defaults to 30 seconds.
- Node MP/POOLING wait loops use 30-second timeout per layer/iteration.
- Practical payload limit is about 255 bytes per XBee packet; GRAPH sender logs warnings when payload exceeds this threshold.
- There is no explicit application-level fragmentation/chunking layer in `zigbee_comm.py`; keep payloads compact.

### Implementation Gotchas to Preserve
- Handlers are dynamically invoked with either `(msg)` or `(msg, xbee_message)` based on callback signature.
- Central `send_to_node()` caches publishers by `(node_name, topic)`; maintain this behavior for throughput.
- If editing handshake transport, update both encode and decode sides consistently, including broadcast discovery handling.

## Development Pitfalls
- Do not assume graph data exists before `GRAPH` is received; `Node.get_neighbors()` gates readiness via `graph_lock`.
- If local subgraph becomes disconnected, node raises a fatal runtime error.
- Hardware-dependent code paths (serial/XBee device, LED matrix) may fail in environments without device access; avoid running them automatically in CI-like contexts unless explicitly requested.

## Change Guidance for AI Agents
- Prefer modifications in `zigbee_comm.py`, `node.py`, and `graph_server.py` that keep protocol compatibility.
- Any schema change to `HandshakeMessage`, `GraphMessage`, or `DataExchangeMessage` must update both serialization (`to_payload`) and deserialization (`from_payload`) paths.
- When changing message contents, verify all producers and consumers:
  - central send path
  - node receive handlers
  - node send path
  - neighbor receive handlers
- For protocol changes, document the updated flow in this file and include migration notes in PR description.
