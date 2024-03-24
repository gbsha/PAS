import unittest
from pycm.fec import Repetition, FEC
from pycm import sim


class Test_sim(unittest.TestCase):
    
    def test_mc(self):
        n = 3
        k = 1
        rep = Repetition(n=n)
        fec = FEC(Repetition.paritychecker(n))

        results = sim.mc(encode=rep.encode, 
                         decode=fec.decode_sd, 
                         k=k, 
                         min_nwe=20,
                         max_time=1, 
                         SNRdB=0, 
                         decision='HD')
        self.assertTrue(results[0] == results[2])
        self.assertTrue(results[1] == results[3])

    def test_campaign(self):
        n = 3
        k = 1
        rep = Repetition(n=n)
        fec = FEC(Repetition.paritychecker(n))

        results = sim.campaign(encode=rep.encode, 
                               decode=fec.decode_sd, 
                               k=k, 
                               min_nwe=20,
                               max_time=2, 
                               SNRdBs=[0, 1], 
                               decision='HD')
        for result in zip(*results):
            self.assertTrue(result[0] == result[2])
            self.assertTrue(result[1] == result[3])


if __name__ == '__main__':
    unittest.main()                
    