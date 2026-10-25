import unittest

from scripts.resolve_data_protocol import buffered_assignment, content_components, quarantine_components


class ProtocolGuardTests(unittest.TestCase):
    def test_buffers_reserve_all_images_without_overlap(self):
        records = [{"id": str(i)} for i in range(1015)]
        active, reserve = buffered_assignment(records, 20)
        self.assertEqual(len(reserve), 40)
        self.assertEqual(len(active), 975)
        self.assertFalse(set(active)&set(reserve))
        self.assertEqual(set(active)|set(reserve), {r['id'] for r in records})
        for left,right in [('train','val'),('val','test')]:
            self.assertEqual(min(int(k) for k,v in active.items() if v==right)-max(int(k) for k,v in active.items() if v==left),21)

    def test_transitive_similarity_is_quarantined_across_sets(self):
        # 1--2 and 2--3 connect; 1--3 alone is above the threshold.
        components=content_components(['1','2','3','4'],{'1':0,'2':1,'3':3,'4':255},threshold=1)
        self.assertEqual(components['1'],components['3'])
        active,reserve=quarantine_components({'1':'train','2':'val','3':'test','4':'train'},{},components)
        self.assertEqual(active,{'3':'test','4':'train'})
        self.assertEqual(set(reserve),{'1','2'})


if __name__=='__main__':
    unittest.main()
