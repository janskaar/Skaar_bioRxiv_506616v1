TITLE Static delta synapse

COMMENT 
Synaptic current is set to weight for 0.1 ms
ENDCOMMENT

DEFINE MAX_SPIKES 1000
DEFINE CUTOFF 20

NEURON {
	POINT_PROCESS UnitISyn
	RANGE tau, i, q
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
}


ASSIGNED {
	i (nA)
	q
	onset_times[MAX_SPIKES] (ms)
	weight_list[MAX_SPIKES] (nA)
}

INITIAL {
	i  = 0
	q  = 0 : queue index
}

BREAKPOINT {
	LOCAL k, expired_spikes, x
	i = 0
	expired_spikes = 0
	FROM k=0 TO q-1 {
		x = (t - onset_times[k])
		if (x > CUTOFF) {
			expired_spikes = expired_spikes + 1
		} else {
            if (x < 0.1) {
                i = weight_list[k]
            } else {
                i = 0
            }
		}
	}
	update_queue(expired_spikes)
}

FUNCTION update_queue(n) {
	LOCAL k
	:if (n > 0) { printf("Queue changed. t = %4.2f onset_times=[",t) }
	FROM k=0 TO q-n-1 {
		onset_times[k] = onset_times[k+n]
		weight_list[k] = weight_list[k+n]
		:if (n > 0) { printf("%4.2f ",onset_times[k]) }
	}
	:if (n > 0) { printf("]\n") }
	q = q-n
}

NET_RECEIVE(weight (nA)) {
	onset_times[q] = t
	weight_list[q] = weight
	if (q >= MAX_SPIKES-1) {
		printf("Error in UnitISyn. Spike queue is full\n")
	} else {
		q = q + 1
	}
}
