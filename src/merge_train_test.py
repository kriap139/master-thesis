from Util import Dataset, Builtin
import gc

def check_merge(b: Builtin):
    shapes = []
    for d in (Dataset(b), Dataset(b, is_test=True), Dataset(b, force_no_train_test=True)):
        frame = d.load_frame()
        shapes.append(frame.shape)

        print(frame.info())
        print()
        
        del frame
        gc.collect()

    train_shape, test_shape, shape = shapes
    print(f"train_shape={train_shape}, test_shape={test_shape}, shape={shape}")

    assert (train_shape[0] + test_shape[0]) == shape[0]
    assert train_shape[1] == shape[1]

def merge(b: Builtin):
    Dataset.merge_train_test(b)

if __name__ == "__main__":
    b = Builtin.EPSILON
    # merge(b)
    check_merge(b)
