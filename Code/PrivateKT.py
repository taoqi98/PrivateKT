import numpy as np 
from keras.utils.np_utils import to_categorical

def calculate_beta(epsilon,K,num_classes):
    beta = (np.exp(epsilon/K)-1)/(np.exp(epsilon/K)-1+num_classes)
    return beta

def LDP(pl,beta,num_classes):
    size = len(pl)
    pl = to_categorical(pl,num_classes=num_classes)

    # Draw Bernoulli random variable for each local prediction
    x_c = np.random.uniform(0,1,size)
    x_c = np.array(x_c<beta).reshape((len(x_c),1))
    
    # Generate random category label for each local prediction 
    n_c = np.random.randint(0,num_classes,size)
    n_c = to_categorical(n_c,num_classes=num_classes)
    
    # Perturb
    pl = x_c*pl + (1-x_c)*n_c
    return pl

class KnowledgeBuffer():
    def __init__(self,size):
        self.size = size
        self.index = []
        self.predits = []

    def push(self,index,predits):
        # Store aggergated knowledge in the current round
        self.index.append(index)
        self.predits.append(predits)
        
        # Make sure the buffer contains at most ``size'' pieces of aggergated kwnoeldge
        self.index = self.index[-self.size:]
        self.predits = self.predits[-self.size:]
        
    def fetch(self,):
        # Read knowledge in the buffer
        return np.concatenate(self.index,axis=0),np.concatenate(self.predits,axis=0)

def importance_sampling(k,model,public_images):
    # Information entropy
    coff = model.predict(public_images)
    coff = (-coff*np.log(coff+10**(-8))).sum(axis=-1)
    
    # Sampling probability
    prob = np.exp(coff)/np.exp(coff).sum()

    # Sampling knolwedge transfer data
    selections = np.random.choice(len(public_images),k,replace=False,p=prob)
    return selections

def PrivateKT(knowledge_buffer,model,train_users,train_images,train_labels,public_images,beta,k,epoch1=5,epoch2=4,self_train_ratio=0.1,sampled_user_ratio=0.5):
    num_classes = train_labels.shape[1]
    uploaded_perturbed_predictions = []

    last_weights = model.get_weights()        
    knowledge_transfer_data = importance_sampling(k,model,public_images)

    # Sample users for model training
    sampled_user_num = int(len(train_users)*sampled_user_ratio)
    sampled_users = np.random.permutation(len(train_users))[:sampled_user_num]
    
    # For each sampled user, do:
    for uid in sampled_users:
        # Local model training
        sample_indexs = train_users[uid]
        x = train_images[sample_indexs]
        y = train_labels[sample_indexs]
        for j in range(epoch1):
            model.fit(x,y,verbose=False)
        
        # Computing local model predictions
        local_predictions = model.predict(public_images[knowledge_transfer_data])
        local_predictions = local_predictions.argmax(axis=-1)

        # Perturb local model predictions
        perturbed_local_predictions = LDP(local_predictions,beta,num_classes)

        # Upload perturbed local model predictions
        uploaded_perturbed_predictions.append(perturbed_local_predictions)
        model.set_weights(last_weights)
                
    # Knolwedge aggergation
    uploaded_perturbed_predictions = np.array(uploaded_perturbed_predictions)
    aggregated_predictions = (uploaded_perturbed_predictions.mean(axis=0)-(1-beta)/num_classes)/beta
    aggregated_predictions = aggregated_predictions.argmax(axis=-1) 
    aggregated_predictions = to_categorical(aggregated_predictions,num_classes=num_classes)
    
    # Store aggergated knolwedge in the knowledge buffer
    knowledge_buffer.push(knowledge_transfer_data,aggregated_predictions)
    
    # Fine-tune the global model on the knowledge buffer
    x,y = knowledge_buffer.fetch()
    for j in range(epoch2):
        model.train_on_batch(public_images[x],y)    
        
    # Self-training
    self_predictions = model.predict(public_images)
    entropy = (-self_predictions*np.log(self_predictions+10**(-8))).sum(axis=-1)
    prob = np.exp(-entropy)/np.exp(-entropy).sum()
    self_train_samples = np.random.choice(len(public_images),int(len(public_images)*self_train_ratio),replace=False,p=prob)
    model.fit(public_images[self_train_samples],self_predictions[self_train_samples],verbose=False)
        