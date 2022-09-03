import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *
from .BaGTI.src.utils import *

class GOBIScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		#model-resnet-migration-bisector
		self.model = eval("energy_latency31_50(1, ResBlock, outputs=4)")

		#model-cnvolution-migration-bisector
		#self.model = eval("energy_latency6_50()")

		#model-cnvolution-orginal gobi :)
		#self.model = eval("energy_latency_50()")


		        # ejra roye GPUuuuuuuuuuuuuuuuuuuuuuuuuu

		self.model=self.model.cuda()
		#self.model=self.model
		#model-cnvolution-migration-bisector
		#self.model, _, _, _ = load_model('energy_latencybmr_50', self.model, data_type)
		#moodel-resnet-migration-bisector
		self.model, _, _, _ = load_model('energy_latencyr_50', self.model, data_type)
		#model-cnvolution-orginal gobi :)
		#self.model, _, _, _ = load_model('energy_latency_50', self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		dtl = data_type.split('_')
		#model-cnvolution-migration-bisector and resnet-migration-bisector
		_, _, self.max_container_ips , self.max_ram_container,self.max_migrat = eval("load_energy_latency_r_data(50)")
		# _, _, self.max_container_ips, self.max_ram_container = load_energy_latency3_data(50)
		#model-cnvolution-orginal gobi :)mmmmmmmmmmmmmmm
		#_, _, self.max_container_ips = load_energy_latency_data(50)
	def run_GOBI(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		if 'latency' in self.model.name:
			cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
			cpuC = np.array([cpuC]).transpose()
			cpu = np.concatenate((cpu, cpuC), axis=1)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result, iteration, fitness = opt(init, self.model, [], self.data_type)
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision


	def newrun_GOBI(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		if 'latency' in self.model.name:
			cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
			cpuC = np.array([cpuC]).transpose()
			ramC = [(c.getRAM()[0]/self.max_ram_container if c else 0) for c in self.env.containerlist]
			ramC = np.array([ramC]).transpose()
			cpu = np.concatenate((cpu, cpuC,ramC), axis=1)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result, iteration, fitness = opt(init, self.model, [], self.data_type)
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision

	# m1 resnet ba migration va nimsaz
	def newrun_GOBI_r(self):
			cpu = [host.getCPU()/100 for host in self.env.hostlist]
			cpu = np.array([cpu]).transpose()
			if 'latency' in self.model.name:
				cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
				cpuC = np.array([cpuC]).transpose()
				ramC = [(c.getRAM()[0]/self.max_ram_container if c else 0) for c in self.env.containerlist]
				ramC = np.array([ramC]).transpose()
				cpu = np.concatenate((cpu, cpuC,ramC), axis=1)
			alloc = []; prev_alloc = {}
			for c in self.env.containerlist:
				oneHot = [0] * len(self.env.hostlist)
				if c: prev_alloc[c.id] = c.getHostID()
				if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
				else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
				alloc.append(oneHot)
			init = np.concatenate((cpu, alloc), axis=1)
			init = torch.tensor(init, dtype=torch.float, requires_grad=True)
			result, iteration, fitness = opt(init, self.model, [], self.data_type)
			decision = []
			for cid in prev_alloc:
				one_hot = result[cid, -self.hosts:].tolist()
				new_host = one_hot.index(max(one_hot))
				if prev_alloc[cid] != new_host: decision.append((cid, new_host))
			return decision


	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.newrun_GOBI_r()
		##decision = self.run_GOBI()
		return decision



class OGOBIScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		#model-resnet-migration-bisector
		#self.model = eval("energy_latency31_50(1, ResBlock, outputs=4)")

		#model-cnvolution-migration-bisector
		#self.model = eval("energy_latency6_50()")

		#model-cnvolution-orginal gobi :)
		self.model = eval("energy_latency_50()")


		        # ejra roye GPUuuuuuuuuuuuuuuuuuuuuuuuuu

		# self.model=self.model.cuda()
		self.model=self.model
		#model-cnvolution-migration-bisector
		#self.model, _, _, _ = load_model('energy_latencybmr_50', self.model, data_type)
		
		#model-cnvolution-orginal gobi :)
		self.model, _, _, _ = load_model('energy_latency_50', self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		dtl = data_type.split('_')
		#model-cnvolution-migration-bisector and resnet-migration-bisector
		#_, _, self.max_container_ips , self.max_ram_container,self.max_migrat = eval("load_energy_latency_r_data(50)")
		# _, _, self.max_container_ips, self.max_ram_container = load_energy_latency3_data(50)
		#model-cnvolution-orginal gobi :)
		_, _, self.max_container_ips = load_energy_latency_data(50)
	def run_GOBI(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		if 'latency' in self.model.name:
			cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
			cpuC = np.array([cpuC]).transpose()
			cpu = np.concatenate((cpu, cpuC), axis=1)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result, iteration, fitness = opt(init, self.model, [], self.data_type)
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision


	def newrun_GOBI(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		if 'latency' in self.model.name:
			cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
			cpuC = np.array([cpuC]).transpose()
			ramC = [(c.getRAM()[0]/self.max_ram_container if c else 0) for c in self.env.containerlist]
			ramC = np.array([ramC]).transpose()
			cpu = np.concatenate((cpu, cpuC,ramC), axis=1)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result, iteration, fitness = opt(init, self.model, [], self.data_type)
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision

	# m1 resnet ba migration va nimsaz
	def newrun_GOBI_r(self):
			cpu = [host.getCPU()/100 for host in self.env.hostlist]
			cpu = np.array([cpu]).transpose()
			if 'latency' in self.model.name:
				cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
				cpuC = np.array([cpuC]).transpose()
				ramC = [(c.getRAM()[0]/self.max_ram_container if c else 0) for c in self.env.containerlist]
				ramC = np.array([ramC]).transpose()
				cpu = np.concatenate((cpu, cpuC,ramC), axis=1)
			alloc = []; prev_alloc = {}
			for c in self.env.containerlist:
				oneHot = [0] * len(self.env.hostlist)
				if c: prev_alloc[c.id] = c.getHostID()
				if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
				else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
				alloc.append(oneHot)
			init = np.concatenate((cpu, alloc), axis=1)
			init = torch.tensor(init, dtype=torch.float, requires_grad=True)
			result, iteration, fitness = opt(init, self.model, [], self.data_type)
			decision = []
			for cid in prev_alloc:
				one_hot = result[cid, -self.hosts:].tolist()
				new_host = one_hot.index(max(one_hot))
				if prev_alloc[cid] != new_host: decision.append((cid, new_host))
			return decision


	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		#decision = self.newrun_GOBI_r()
		decision = self.run_GOBI()
		return decision