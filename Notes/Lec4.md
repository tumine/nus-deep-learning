process identification: IP address of the host + **transport layer protocol** + port number (of the host)
- transport layer protocol is determined by protocol of higher layers
- a port number can be reused in case that different protocols are adopted
- host IP is allocated by DHCP in the same network

socket: interface to access the transport layer protocol managed by OS

socket programming with TCP
- client process initiates a connection through a randomly chosen port number
- at server site, the welcome socket at port 80 **duplicates a new socket** for the incoming connection request (still through port 80)
  - the client port number can't be reused, since a client can only connect to a single server
- connection identification: Dst IP/Port, Src IP/Port
- once the connection is established, the concept of client/server won't matter

higher level protocols of socket connection: WebSockets, MQTT

MQTT
- a central server (Broker) manages all the client messages, and send specific topics of messages to clients who subscribe to them
- asynchronous: senders and receivers needn't be online or connected at the same time

asyncio: **maintain separate tasks** in the same program instead of executing the codes sequentially

type of cryptography
- symmetric: both sides use the same key
  - requires the secure passing of encrypting keys in advance
- asymmetric: sender and receiver use different keys
- public key cryptography
  - the sender uses **the receiver's public key** to encrypt the message
  - the receiver uses his private key to decrypt the message
  - in practice, priorly passing a symmetric session key by public key encryption, then uses the session key to send the main messages (transfer faster with symmetric keys)

---

re-entrant lock (`threading.RLock`): allow a thread to acquire the same lock multiple times without causing a deadlock

conditional wait (`threading.Condition`): a kind of lock that allow waiting when it is held (release the lock when waiting)

synchronization (`threading.Barrier`): forces a number of threads to restart their execution till all of them reaches the specific boundary line
