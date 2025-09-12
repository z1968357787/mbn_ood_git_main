import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cum_mask(P, t, model, p_mask):
    """ 
        Keep track of mask values. 
        This will be used later as a regularizer in the optimization
    """
    try:
        model = model.module
    except AttributeError:
        model = model

    task_id = torch.tensor([t]).cuda()
    mask = {}
    for n, _ in model.named_parameters():
        names = n.split('.')
        checker = [i for i in ['ec0', 'ec1', 'ec2'] if i in names]
        if names[0] == 'module':
            names = names[1:]
        if checker:
            if 'layer' in n:
                gc1, gc2 = model.__getattr__(names[0])[int(names[1])].mask(task_id, s=P.epochs)
                if checker[0] == 'ec1':
                    n = '.'.join(n.split('.')[:-1]) # since n is like layer2.0.ec1.8, where the last number 8 indicates task id
                    mask[n] = gc1.detach()
                    mask[n].requires_grad = False
                elif checker[0] == 'ec2':
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = gc2.detach()
                    mask[n].requires_grad = False
                # elif 'down_sample' in n:
                #     mask[n] = self.model.__getattr__(names[0]).down_sample.mask(t, s=self.epochs).detach()
                #     mask[n].requires_grad = False

            elif checker[0] == 'ec0':
                n = '.'.join(n.split('.')[:-1])
                mask[n] = model.mask(task_id, P.epochs).detach()
                mask[n].requires_grad = False

    if p_mask is None:
        p_mask = {}
        for n in mask.keys():
            p_mask[n] = mask[n]
    else:
        for n in mask.keys():
            p_mask[n] = torch.max(p_mask[n], mask[n])
    return p_mask

def more_freeze_mask(self, t, p_mask):
    """
        Eq (2) in the paper. self.mask_back is a dictionary whose keys are
        the convolutions' parameter names. Each value of a key is a matrix, whose elements are
        approximately binary.

        For ViT Adapter, there's only ec1 and ec2 in adapters in tranformer blocks.
        There are no other ec

        p_mask.keys() are [
        'blocks.0.adapter1.ec1', 'blocks.0.adapter1.ec2',
        'blocks.0.adapter2.ec1', 'blocks.0.adapter2.ec2',
        'blocks.1.adapter1.ec1', 'blocks.1.adapter1.ec2',
        'blocks.1.adapter2.ec1', 'blocks.1.adapter2.ec2',
        'blocks.2.adapter1.ec1', 'blocks.2.adapter1.ec2',
        'blocks.2.adapter2.ec1', 'blocks.2.adapter2.ec2',
        'blocks.3.adapter1.ec1', 'blocks.3.adapter1.ec2',
        'blocks.3.adapter2.ec1', 'blocks.3.adapter2.ec2',
        'blocks.4.adapter1.ec1', 'blocks.4.adapter1.ec2',
        'blocks.4.adapter2.ec1', 'blocks.4.adapter2.ec2',
        'blocks.5.adapter1.ec1', 'blocks.5.adapter1.ec2',
        'blocks.5.adapter2.ec1', 'blocks.5.adapter2.ec2',
        'blocks.6.adapter1.ec1', 'blocks.6.adapter1.ec2',
        'blocks.6.adapter2.ec1', 'blocks.6.adapter2.ec2',
        'blocks.7.adapter1.ec1', 'blocks.7.adapter1.ec2',
        'blocks.7.adapter2.ec1', 'blocks.7.adapter2.ec2',
        'blocks.8.adapter1.ec1', 'blocks.8.adapter1.ec2',
        'blocks.8.adapter2.ec1', 'blocks.8.adapter2.ec2',
        'blocks.9.adapter1.ec1', 'blocks.9.adapter1.ec2',
        'blocks.9.adapter2.ec1', 'blocks.9.adapter2.ec2',
        'blocks.10.adapter1.ec1', 'blocks.10.adapter1.ec2',
        'blocks.10.adapter2.ec1', 'blocks.10.adapter2.ec2',
        'blocks.11.adapter1.ec1', 'blocks.11.adapter1.ec2',
        'blocks.11.adapter2.ec1', 'blocks.11.adapter2.ec2'
        ]
    """
    try:
        self.net = self.net.module
    except AttributeError:
        self.net = self.net

    mask_back = {}
    for n, p in self.net.named_parameters():
        names = n.split('.')
        if 'adapter' in n: # adapter1 or adapter2. adapter.ec1, adapter.ec2
            # e.g. n is blocks.1.adapter1.fc1.weight
            if 'fc1.weight' in n:
                mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec1'].data.view(-1, 1).expand_as(p)
            elif 'fc1.bias' in n:
                mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec1'].data.view(-1)
            elif 'fc2.weight' in n:
                post = p_mask['.'.join(names[:-2]) + '.ec2'].data.view(-1, 1).expand_as(p)
                pre  = p_mask['.'.join(names[:-2]) + '.ec1'].data.view(1, -1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif 'fc2.bias' in n:
                mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec2'].view(-1)
    return mask_back

def wptp_freeze_mask(model, p_mask): #P其实不需要传入
    """
        Eq (2) in the paper. self.mask_back is a dictionary whose keys are
        the convolutions' parameter names. Each value of a key is a matrix, whose elements are
        approximately binary.
    """
    try:
        model = model.module
    except AttributeError:
        model = model

    mask_back = {}
    for n, p in model.named_parameters():
        names = n.split('.')
        if 'layer' not in names[0]:
            if n == 'conv1.weight':
                mask_back[n] = 1 - p_mask['ec0'].data.view(-1, 1, 1, 1).expand_as(p)
        # elif 'layer' in names[0]:
        elif 'layer1' in n:
            if n == 'layer1.0.conv1.weight':
                post = p_mask['layer1.0.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['ec0'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer1.0.conv2.weight':
                post = p_mask['layer1.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer1.0.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer1.1.conv1.weight':
                post = p_mask['layer1.1.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer1.0.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer1.1.conv2.weight':
                post = p_mask['layer1.1.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer1.1.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

        elif 'layer2' in n:
            if n == 'layer2.0.conv1.weight':
                post = p_mask['layer2.0.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer1.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer2.0.conv2.weight':
                post = p_mask['layer2.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer2.0.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer2.0.shortcut.conv1.weight':
                post = p_mask['layer2.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer1.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer2.1.conv1.weight':
                post = p_mask['layer2.1.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer2.0.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer2.1.conv2.weight':
                post = p_mask['layer2.1.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer2.1.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

        elif 'layer3' in n:
            if n == 'layer3.0.conv1.weight':
                post = p_mask['layer3.0.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer2.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer3.0.conv2.weight':
                post = p_mask['layer3.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer3.0.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer3.0.shortcut.conv1.weight':
                post = p_mask['layer3.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer2.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer3.1.conv1.weight':
                post = p_mask['layer3.1.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer3.0.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer3.1.conv2.weight':
                post = p_mask['layer3.1.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer3.1.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

        elif 'layer4' in n:
            if n == 'layer4.0.conv1.weight':
                post = p_mask['layer4.0.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer3.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer4.0.conv2.weight':
                post = p_mask['layer4.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer4.0.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer4.0.shortcut.conv1.weight':
                post = p_mask['layer4.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer3.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer4.1.conv1.weight':
                post = p_mask['layer4.1.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer4.0.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer4.1.conv2.weight':
                post = p_mask['layer4.1.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = p_mask['layer4.1.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
    return mask_back

def cum_mask_simclr(P, t, model, p_mask):
    assert p_mask is not None

    try:
        model = model.module
    except AttributeError:
        model = model

    task_id = torch.tensor([t]).cuda()

    mask = {}
    for n, _ in model.named_parameters():
        if 'simclr_layer.ec' in n:
            n = '.'.join(n.split('.')[:-1])
            gc1 = model.simclr_layer.mask(task_id, s=P.epochs)
            mask[n] = gc1.detach()
            mask[n].requires_grad = False

    if 'simclr_layer.ec' in p_mask.keys():
        for n in mask.keys():
            p_mask[n] = torch.max(p_mask[n], mask[n])
    else:
        for n in mask.keys():
            p_mask[n] = mask[n]
    return p_mask

def freeze_mask_simclr(P, t, model, p_mask, mask_back):
    assert mask_back is not None

    try:
        model = model.module
    except AttributeError:
        model = model

    for n, p in model.named_parameters():
        if n == 'simclr_layer.fc1.weight':
            post = p_mask['simclr_layer.ec'].data.view(-1, 1).expand_as(p)
            pre  = p_mask['layer4.1.ec2'].data.view(-1, 1, 1).expand((p_mask['layer4.1.ec2'].size(1), 1, 1)).contiguous().view(1, -1).expand_as(p)
            mask_back[n] = 1 - torch.min(post, pre)
        elif n == 'simclr_layer.fc1.bias':
            mask_back[n] = 1 - p_mask['simclr_layer.ec'].data.view(-1)
    return mask_back
