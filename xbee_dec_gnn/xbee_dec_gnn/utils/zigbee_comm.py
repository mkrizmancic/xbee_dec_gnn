import inspect
import logging
import pickle
import random
import threading
import time
from enum import Enum

from digi.xbee.devices import ZigBeeDevice
from digi.xbee.exception import TimeoutException, TransmitException
from digi.xbee.models.address import XBee16BitAddress, XBee64BitAddress


class Topic(Enum):
    HANDSHAKE = 1
    GRAPH = 2
    MP = 3
    POOLING = 4


class HandshakeStep(Enum):
    DISCOVERY = 1
    REGISTER = 2
    ACK = 3
    CONFIRM = 4


class MessageBase:
    def __init__(self, topic: Topic):
        self.topic = topic  # TODO: remove topic

    def to_payload(self) -> dict:
        return {}

    def summary(self) -> str:
        return self.topic.name

    def to_bytes(self, topic_override: Topic | None = None) -> bytes:
        topic = topic_override or self.topic
        if topic != self.topic:
            raise ValueError(f"Topic mismatch: message={self.topic} publisher={topic}")
        payload = pickle.dumps(self.to_payload(), protocol=5)
        return bytes([topic.value]) + payload


class HandshakeMessage(MessageBase):
    def __init__(
        self,
        step: HandshakeStep,
        sender_name: str | None = None,
        sender_addr: str | None = None,
        addr_map: dict | None = None,
    ):
        super().__init__(Topic.HANDSHAKE)
        self.step = step
        self.sender_name = sender_name
        self.sender_addr = sender_addr
        self.addr_map = addr_map

    def to_payload(self) -> dict:
        return {
            "step": self.step.value,
            "sender_name": self.sender_name,
            "sender_addr": self.sender_addr,
            "addr_map": self.addr_map,
        }

    def summary(self) -> str:
        return f"HANDSHAKE:{self.step.name}"

    @classmethod
    def from_payload(cls, payload: dict) -> "HandshakeMessage":
        return cls(
            step=HandshakeStep(payload["step"]),
            sender_name=payload["sender_name"],
            sender_addr=payload["sender_addr"],
            addr_map=payload["addr_map"],
        )


class GraphMessage(MessageBase):
    def __init__(self, neighbors: list[str], features: list[float]):
        super().__init__(Topic.GRAPH)
        self.neighbors = neighbors
        self.features = features

    def to_payload(self) -> dict:
        return {
            "neighbors": self.neighbors,
            "features": self.features,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "GraphMessage":
        return cls(
            neighbors=payload.get("neighbors") or [],
            features=payload.get("features") or [],
        )


class DataExchangeMessage(MessageBase):
    def __init__(
        self,
        topic: Topic,
        sender_name: str,
        layer: int | None = None,
        round_id: int | None = None,
        iteration: int | None = None,
        data: bytes | None = None,
        shape: list[int] | None = None,
        sources: list | None = None,
    ):
        if topic not in (Topic.MP, Topic.POOLING):
            raise ValueError("DataExchangeMessage requires topic MP or POOLING")
        super().__init__(topic)
        self.sender_name = sender_name
        self.layer = layer
        self.round_id = round_id
        self.iteration = iteration
        self.data = data
        self.shape = shape
        self.sources = sources

    @classmethod
    def for_message_passing(
        cls,
        sender_name: str,
        layer: int,
        round_id: int,
        data: bytes,
        shape: list[int],
    ) -> "DataExchangeMessage":
        return cls(
            topic=Topic.MP,
            sender_name=sender_name,
            layer=layer,
            round_id=round_id,
            data=data,
            shape=shape,
        )

    @classmethod
    def for_pooling(
        cls,
        sender_name: str,
        iteration: int,
        data: bytes,
        shape: list[int],
        sources: list | None = None,
    ) -> "DataExchangeMessage":
        return cls(
            topic=Topic.POOLING,
            sender_name=sender_name,
            iteration=iteration,
            data=data,
            shape=shape,
            sources=sources,
        )

    def to_payload(self) -> dict:
        return {
            "sender_name": self.sender_name,
            "layer": self.layer,
            "round_id": self.round_id,
            "iteration": self.iteration,
            "data": self.data,
            "shape": self.shape,
            "sources": self.sources,
        }

    @classmethod
    def from_payload(cls, topic: Topic, payload: dict) -> "DataExchangeMessage":
        return cls(
            topic=topic,
            sender_name=payload["sender_name"],
            layer=payload.get("layer"),
            round_id=payload.get("round_id"),
            iteration=payload.get("iteration"),
            data=payload.get("data"),
            shape=payload.get("shape"),
            sources=payload.get("sources"),
        )


def decode_message(data: bytes) -> MessageBase:
    if not data:
        raise ValueError("Message payload too short")

    topic = Topic(data[0])
    payload = {}
    if len(data) > 1:
        payload = pickle.loads(data[1:])

    if topic == Topic.HANDSHAKE:
        return HandshakeMessage.from_payload(payload)
    if topic == Topic.GRAPH:
        return GraphMessage.from_payload(payload)
    if topic in (Topic.MP, Topic.POOLING):
        return DataExchangeMessage.from_payload(topic, payload)

    raise ValueError(f"Unknown topic: {topic}")


class XBeePublisher:
    def __init__(self, zigbee_device, target_addr, target_name, topic: Topic, logger=None):
        self.device = zigbee_device
        self.target_name = target_name
        self.topic = topic

        if isinstance(target_addr, str):
            self.target_addr = XBee64BitAddress.from_hex_string(target_addr)
        else:
            self.target_addr = target_addr

        self.logger = logger or logging.getLogger(__name__)

    def publish(self, msg: MessageBase, add_random_delay: bool = False) -> bool:
        if add_random_delay:
            time.sleep(random.uniform(0.05, 0.2))

        data = msg.to_bytes(topic_override=self.topic)

        ok = False
        backoff_delay = 0.05
        for attempt in range(1, 5):
            try:
                self.device.send_data_64_16(self.target_addr, XBee16BitAddress.UNKNOWN_ADDRESS, data)
                ok = True
                if attempt == 1:
                    self.logger.debug(f"TX: {msg.summary()} -> {self.target_name}")
                else:
                    self.logger.debug(
                        f"TX: {msg.summary()} -> {self.target_name} (retry {attempt})"
                    )
                break
            except (TransmitException, TimeoutException) as exc:
                status = getattr(exc, "transmit_status", None) or getattr(exc, "status", None)
                self.logger.warning(
                    f"TX fail: {msg.summary()} (attempt {attempt}, {status})"
                )
                if attempt < 4:
                    time.sleep(backoff_delay)
                    backoff_delay *= 2

        if not ok:
            self.logger.error(f"TX gave up: {msg.summary()} to {self.target_name}")

        time.sleep(0.05)

        return ok


class ZigbeeInterfaceBase:
    def __init__(self, port, baud_rate, logger=None):
        self.port = port
        self.baud_rate = baud_rate
        self.network = None
        self.logger = logger or logging.getLogger(__name__)

        self.device = ZigBeeDevice(self.port, self.baud_rate)
        self._handlers = {}

    def register_handler(self, topic: Topic, callback):
        if not isinstance(topic, Topic):
            raise TypeError(f"topic must be a Topic enum, got {type(topic)}")
        self._handlers[topic] = callback
        self.logger.debug(f"Registered handler for {topic.name}")

    def start(self):
        self.device.open()
        self.device.add_data_received_callback(self._data_receive_callback)
        self.network = self.device.get_network()

    def _decode_message(self, payload: bytes) -> MessageBase | None:
        try:
            return decode_message(payload)
        except Exception as exc:
            self.logger.warning(f"Failed to decode message: {exc}")
            return None

    def _data_receive_callback(self, xbee_message):
        msg = self._decode_message(xbee_message.data)
        if msg is None:
            return

        handler = self._handlers.get(msg.topic)
        if handler is None:
            self.logger.warning(f"No handler registered for topic: {msg.topic.name}")
            return

        try:
            self._invoke_handler(handler, msg, xbee_message)
        except Exception as exc:
            self.logger.error(
                f"Error in handler for {msg.topic.name}: {exc}", exc_info=True
            )

    def _handler_param_count(self, handler):
        try:
            sig = inspect.signature(handler)
            count = len(sig.parameters)
            if inspect.ismethod(handler):
                count -= 1
            return count
        except (TypeError, ValueError):
            return 1

    def _invoke_handler(self, handler, msg, xbee_message):
        if self._handler_param_count(handler) >= 2:
            handler(msg, xbee_message)
        else:
            handler(msg)

    def _create_publisher_for_addr(self, target_addr, target_name, topic: Topic, logger=None):
        return XBeePublisher(
            zigbee_device=self.device,
            target_addr=target_addr,
            target_name=target_name,
            topic=topic,
            logger=logger or self.logger,
        )


def _coerce_node_id(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class ZigbeeNodeInterface(ZigbeeInterfaceBase):
    def __init__(self, port, baud_rate, node_name, num_nodes, logger=None):
        super().__init__(port, baud_rate, logger=logger)

        self.node_name = node_name
        self.num_nodes = num_nodes

        self.addr_map = {}
        self.central_addr = None
        self.df = False
        self._handshake_lock = threading.Lock()
        self._handshake_event = threading.Event()

        self.register_handler(Topic.HANDSHAKE, self._handle_handshake)

    def _handle_handshake(self, msg: HandshakeMessage):
        if msg.step == HandshakeStep.DISCOVERY:
            self.central_addr = msg.sender_addr
            self.logger.info("Handshake: DISCOVERY received from central server")

            response = HandshakeMessage(
                step=HandshakeStep.REGISTER,
                sender_name=self.node_name,
                sender_addr=str(self.device.get_64bit_addr()),
            )
            self._send_handshake_message(response)
            return

        if msg.step == HandshakeStep.ACK:
            self.addr_map = msg.addr_map

            if not self.addr_map:
                self.logger.error("Handshake: received empty addr_map in ACK")
                return

            self.logger.info(f"Handshake: Received {len(self.addr_map)} peer addresses")

            response = HandshakeMessage(step=HandshakeStep.CONFIRM, sender_name=self.node_name)
            self._send_handshake_message(response)

            with self._handshake_lock:
                self._handshake_complete = True
                self._handshake_event.set()

            self.logger.info("Handshake complete: ready for communication")

    def _send_handshake_message(self, msg: HandshakeMessage):
        if self.central_addr is None:
            self.logger.error("Cannot send handshake message: central address unknown")
            return

        handshake_pub = XBeePublisher(
            zigbee_device=self.device,
            target_addr=self.central_addr,
            target_name="CENTRAL",
            topic=Topic.HANDSHAKE,
            logger=self.logger,
        )
        handshake_pub.publish(msg, add_random_delay=False)

    def wait_for_handshake(self, timeout=30.0):
        self.logger.info("Waiting for handshake to complete...")
        success = self._handshake_event.wait(timeout=timeout)

        if success:
                self.logger.info(f"Handshake complete: node_name={self.node_name}")
        else:
            self.logger.error(f"Handshake timeout after {timeout:.1f}s")

        return success

    def create_publisher(self, target_name, topic: Topic, logger=None):
        if target_name not in self.addr_map:
            raise ValueError(f"Unknown target node ID: {target_name}. Handshake may not be complete.")

        target_addr = self.addr_map[target_name]

        return self._create_publisher_for_addr(
            target_addr=target_addr,
            target_name=target_name,
            topic=topic,
            logger=logger,
        )


class ZigbeeCentralInterface(ZigbeeInterfaceBase):
    BCAST_64 = XBee64BitAddress.from_hex_string("000000000000FFFF")
    BCAST_16 = XBee16BitAddress.from_hex_string("FFFE")

    def __init__(
        self,
        port,
        baud_rate,
        num_nodes,
        wait_forever=True,
        init_timeout_s=30.0,
        logger=None,
    ):
        super().__init__(port, baud_rate, logger=logger)

        self.num_nodes = num_nodes
        self.addr_map = {}
        self._publisher_cache = {}

        self._received_register = set()
        self._received_confirm = set()
        self._reg_event = threading.Event()
        self._confirm_event = threading.Event()
        self.wait_forever = wait_forever
        self.init_timeout_s = init_timeout_s
        self._next_auto_id = 0

        self.register_handler(Topic.HANDSHAKE, self._handle_handshake)

    def _assign_node_id(self):
        while self._next_auto_id in self.addr_map:
            self._next_auto_id += 1
        assigned = self._next_auto_id
        self._next_auto_id += 1
        return assigned

    def _handle_handshake(self, msg: HandshakeMessage):
        if msg.step == HandshakeStep.REGISTER:
            self.addr_map[msg.sender_name] = str(msg.sender_addr)

            self.logger.info(
                f"RX: NODE_REGISTER from {msg.sender_name} [{msg.sender_addr}]"
            )

            self._received_register.add(msg.sender_name)
            if len(self._received_register) >= self.num_nodes:
                self._reg_event.set()
            return

        if msg.step == HandshakeStep.CONFIRM:
            self._received_confirm.add(msg.sender_name)

            self.logger.info(f"RX: ID_CONFIRM from {msg.sender_name}")

            if len(self._received_confirm) >= self.num_nodes:
                self._confirm_event.set()

    def run_handshake(self):
        my_addr = str(self.device.get_64bit_addr())

        self.logger.info(f"Central online: port={self.port} baud={self.baud_rate}")
        self.logger.info(f"Central XBee 64-bit addr: {my_addr}")
        self.logger.info("Handshake: broadcasting DISCOVERY; waiting for INIT from nodes")

        msg = HandshakeMessage(step=HandshakeStep.DISCOVERY, sender_addr=my_addr)
        interval_s = 1.0
        start_time = time.time()

        while not self._reg_event.is_set():
            if not self.wait_forever:
                elapsed = time.time() - start_time
                if elapsed >= self.init_timeout_s:
                    raise TimeoutError(
                        f"[CENTRAL] Only received {len(self._received_register)}/{self.num_nodes} INITs"
                    )

            try:
                self.send_broadcast(msg)
                self.logger.debug(f"TX: DISCOVERY broadcast (central_mac={my_addr})")
            except TransmitException as exc:
                status = getattr(exc, "transmit_status", None) or getattr(exc, "status", None)
                self.logger.warning(f"TX: DISCOVERY broadcast failed (status={status})")

            self._reg_event.wait(timeout=interval_s)

        self.logger.info(
            "Handshake: all INIT received; sending ACK_INIT to assign ids; waiting for ACK_ID confirmations"
        )
        # TODO: check
        handshake_publishers = self._build_node_publishers(topic=Topic.HANDSHAKE)

        for node_name in self.addr_map.keys():
            msg = HandshakeMessage(
                step=HandshakeStep.ACK,
                addr_map=self.addr_map,
            )
            handshake_publishers[node_name].publish(msg, add_random_delay=False)

        if not self._confirm_event.is_set():
            if self.wait_forever:
                self._confirm_event.wait()
            else:
                self._confirm_event.wait(timeout=self.init_timeout_s)
                if not self._confirm_event.is_set():
                    raise TimeoutError(
                        f"[CENTRAL] Only received {len(self._received_confirm)}/{self.num_nodes} ID_CONFIRM"
                    )

        self.logger.info("Handshake: all ID_CONFIRM confirmations received")

    def _build_node_publishers(self, topic: Topic):
        publishers = {}
        for node_id, addr in self.addr_map.items():
            if addr is None:
                self.logger.warning(
                    f"Handshake: addr for node_id={node_id} is None; skipping publisher creation"
                )
                continue

            publishers[node_id] = self._create_publisher_for_addr(
                target_addr=addr,
                target_name=node_id,
                topic=topic,
            )
        return publishers

    def send_to_node(self, node_name, msg: MessageBase, add_random_delay=False):
        if node_name not in self.addr_map:
            self.logger.error(f"No address for node_name={node_name}")
            return False

        key = (node_name, msg.topic)
        publisher = self._publisher_cache.get(key)
        if publisher is None:
            publisher = self._create_publisher_for_addr(
                target_addr=self.addr_map[node_name],
                target_name=node_name,
                topic=msg.topic,
            )
            self._publisher_cache[key] = publisher

        return publisher.publish(msg, add_random_delay=add_random_delay)

    def send_broadcast(self, msg: MessageBase):
        data = msg.to_bytes()
        self.device.send_data_64_16(self.BCAST_64, self.BCAST_16, data)

