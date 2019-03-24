import numpy as np
from sklearn.utils import shuffle

class SiameseLoader(object):
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, data, categories):
        self.data = data
        self.categories = categories
        self.info = {}
        
    def get_batch(self, batch_size, s="train"):
        """
        Create batch of n pairs, half same class, half different class
        
        Parameters
        ----------
        batch_size: int
            Number of pairs to create

        s: str
            Set name

        Returns
        -------
        tuple
            tuple[0] -> array like of shape (batch_size, 2, window_size, n_features, 1) containing the pairs of the batch
            tuple[1] -> array like of shape (batch_size) containing the targets. 
        """
        X = self.data[s]
        y = self.categories[s]
        
        n_seq, seq_length, n_features = X.shape

        # Initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, seq_length, n_features, 1)) for i in range(2)]
        
        # Initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        labels = []
        for i in range(batch_size):
            idx_1 = np.random.randint(0, n_seq)
            pairs[0][i,:,:,:] = X[idx_1].reshape(seq_length, n_features, 1)
            pair_0_label = y[idx_1]
                        
            # Pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                idx_2 = np.random.choice([l[0] for l in enumerate(y) if l[1] == pair_0_label])
            else: 
                idx_2 = np.random.choice([l[0] for l in enumerate(y) if l[1] != pair_0_label])
            
            labels.append((pair_0_label, y[idx_2]))
            
            pairs[1][i,:,:,:] = X[idx_2].reshape(seq_length, n_features, 1)
            
        return pairs, targets, labels
    
    def generate(self, batch_size, s="train"):
        """A generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size, s)
            yield (pairs, targets)    
            
    def make_oneshot_task(self, s="val", N=5):
        """
        Create pairs of test image, support set for testing N way one-shot learning. 
        
        Parametes
        ---------
        s: str, optional
            Name of the used set
        N: int, optional
            Support set size

        Returns
        -------
        tuple
            tuple[0] -> array like of shape (batch_size, 2, window_size, n_features, 1) containing the pairs of the batch
                        Paris where first element is the test image and the second one is an instance of the support set
            tuple[1] -> array like of shape (batch_size) with a single 1, which is the target of support set
        """
        X = self.data[s]
        y = self.categories[s]
        enum_labels = list(enumerate(y))

        n_seq, seq_length, n_features = X.shape
    
        # Pick the true label
        true_label = np.random.choice(y)
        true_instances = np.array([l[0] for l in enum_labels if l[1] == true_label]).astype(np.int)
        false_instances = np.array([l[0] for l in enum_labels if l[1] != true_label]).astype(np.int)

        # Build the support set
        support_set_idx = np.random.choice(false_instances, size=(N))        
        support_set = X[support_set_idx].reshape(N, seq_length, n_features, 1)
        if len(true_instances) == 1:
            test_img_idx, support_true_idx = true_instances[0], true_instances[0]
        else:
            test_img_idx, support_true_idx = np.random.choice(true_instances, size=(2,), replace=False)
        support_set[0,:,:,:] = X[support_true_idx].reshape(seq_length, n_features, 1)
        
        # Pick the same test image N times
        test_img = [X[test_img_idx].reshape(seq_length, n_features, 1)]*N
        
        # Set the first target to 1 because the first element of support set is the desired output
        targets = np.zeros((N,))
        targets[0] = 1
        
        targets, test_img, support_set = shuffle(targets, test_img, support_set)
        pairs = [test_img, support_set]
        
        return pairs, targets
    
    def test_oneshot(self, model, k, s="val", verbose=0):
        """
        Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks
        
        Parameters
        ----------
        model: kearas.model
        
        k: int
            Number of one shot tasks
        s: str, optional
            Name of the set
        verbose: boolean, optional
            If True -> print the accuracy
        
        Returns
        -------
        float
            Accuaracy on the k one shot tasks
        """
        n_correct = 0
        for i in range(k):
            inputs, targets = self.make_oneshot_task(s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% learning accuracy".format(percent_correct))
        return percent_correct
    