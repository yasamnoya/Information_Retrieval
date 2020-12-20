class Data():

    def __init__(self, data_dir):

        doc_list_path = data_dir + "/doc_list.txt"
        query_list_path = data_dir + "/query_list.txt"

        with open(doc_list_path, 'r') as doc_list_file:
            data = doc_list_file.read().strip()
            self.doc_filename_list = [line for line in data.split('\n')]
        pass

        with open(query_list_path, 'r') as query_list_file:
            data = query_list_file.read().strip()
            self.query_filename_list = [line for line in data.split('\n')]
        pass

            s
        self.doc_list = []
        for filename in doc_filename_list:
            path = data_dir + '/docs/' + filename + '.txt'
            doc = open(path , 'r').read().strip().split(' ')
            self.doc_list.append(doc)
        pass

        self.query_list = []
        for filename in query_filename_list:
            path = data_dir + '/queries/' + filename + '.txt'
            query = open(path , 'r').read().strip().split(' ')
            self.query_list.append(query)
        pass
    pass
pass
