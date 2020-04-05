import torch.nn as nn
import torchvision.models as models
from networks import  Wide_ResNet,densenet,resnet,MobileNetV2,distributed_Net

networks={
	'wideresnet-28x10': [28,10],
	'wideresnet-40x10': [40,10],
	'wideresnet-40x14': [40,14],
	'wideresnet-16x8': [16,8],
	'densenet-121': 121,
	'densenet-169': 169,
	'resnet-18': 18,
	'resnet-50': 50,
	'resnet-101': 101
}

def return_network(network_name,pretrained=True,n_classes=None,dropout=None):

	if pretrained:
		if network_name=='resnet-18':
			net = models.resnet18(pretrained=pretrained)		
		elif network_name=='resnet-50':
			net = models.resnet50(pretrained=pretrained)
		elif network_name=='resnet-101':
			net = models.resnet101(pretrained=pretrained)
		elif network_name=='densenet-121':
			net = models.densenet121(pretrained=pretrained)
		elif network_name=='squeezenet1_1':
			net = models.squeezenet1_1(pretrained=pretrained)
		else:
			print ("Provide a valid network, check python main.py -h")
			exit(-1)

		#transfer learning
		if 'densenet' in network_name:
			net.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(net.classifier.in_features,n_classes,bias=True))
		elif 'squeezenet1_1' in network_name:
			net.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1))
			net.num_classes=n_classes
		else:
			net.fc=nn.Sequential(nn.Dropout(dropout),nn.Linear(net.fc.in_features,n_classes,bias=True))
	
	else:
		if 'wideresnet'==network_name.split("-")[0]:
			depth,widenfactor=networks[network_name]
			net=Wide_ResNet(depth,widenfactor,dropout,n_classes)
		if 'densenet'==network_name.split("-")[0]:
			depth=networks[network_name]
			net=densenet(depth,n_classes,dropout)
		if 'resnet'==network_name.split("-")[0]:
			depth=networks[network_name]
			net=resnet(depth,n_classes,dropout)
		if 'mobilenetv2'==network_name:
			net=MobileNetV2(n_classes,dropout)
	
		parameters_fc=list()
		parameters_conv=list()
		for p in net.parameters():
			parameters_fc+=[p]

	return net



def load_network(network_name,pretrained,devices,n_classes=None,dropout=None):

	net = return_network(network_name,pretrained,n_classes,dropout).cuda()
	parameters_conv=list()
	parameters_fc=list()
	if pretrained:
		for p in net.named_parameters():
			if p[0] in ['fc.1.weight','fc.1.bias','classifier.weight','classifier.bias','classifier.1.weight','classifier.1.bias']:
				parameters_fc+=[p[1]]
			else:
				parameters_conv+=[p[1]]	
	else:
		for p in net.parameters():
			parameters_fc+=[p]

	net = distributed_Net(net,devices)

	return net,[parameters_fc,parameters_conv]

	


