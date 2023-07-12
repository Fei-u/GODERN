from vector_fields import *
from GODERN import Model

def make_model(args):
    if args.model_type == 'fc':
        if args.aug_type == 'None':
            vector_field_1 = FinalTanh_f(input_channels=args.input_dim, hidden_channels=2*(args.hid_dim),
                                            hidden_hidden_channels=args.hid_hid_dim,
                                            num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim, is_ode=args.is_ode)
        if args.aug_type == 'z':
            vector_field_1 = FinalTanh_f(input_channels=args.input_dim, hidden_channels=2*(args.hid_dim+5),
                                            hidden_hidden_channels=args.hid_hid_dim,
                                            num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim, is_ode=args.is_ode)
        if args.aug_type == 'fs':
            vector_field_1 = FinalTanh_f(input_channels=args.input_dim, hidden_channels=(args.hid_dim),
                                            hidden_hidden_channels=args.hid_hid_dim,
                                            num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim, is_ode=args.is_ode)
            vector_field_2 = FinalTanh_f(input_channels=args.input_dim, hidden_channels=(args.hid_dim),
                                            hidden_hidden_channels=args.hid_hid_dim,
                                            num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim, is_ode=args.is_ode)
    else:
        if args.aug_type == 'None':
            vector_field_1 = VectorField_g(input_channels=args.input_dim, hidden_channels=2*(args.hid_dim),
                                            hidden_hidden_channels=args.hid_hid_dim,
                                            num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim, is_ode=args.is_ode)
        if args.aug_type == 'z':
            vector_field_1 = VectorField_g(input_channels=args.input_dim, hidden_channels=2*(args.hid_dim+5),
                                            hidden_hidden_channels=args.hid_hid_dim,
                                            num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim, is_ode=args.is_ode)
        if args.aug_type == 'fs':
            vector_field_1 = VectorField_g(input_channels=args.input_dim, hidden_channels=(args.hid_dim),
                                            hidden_hidden_channels=args.hid_hid_dim,
                                            num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim, is_ode=args.is_ode)
            vector_field_2 = VectorField_g(input_channels=args.input_dim, hidden_channels=(args.hid_dim),
                                            hidden_hidden_channels=args.hid_hid_dim,
                                            num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim, is_ode=args.is_ode)
    if args.aug_type == 'fs':
        model = Model(args, func=[vector_field_1,vector_field_2], input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                            output_channels=args.output_dim, rnn_layers=args.rnn_layers,
                                            device=args.device)
    else:
        model = Model(args, func=[vector_field_1,vector_field_1], input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                            output_channels=args.output_dim, rnn_layers=args.rnn_layers,
                                            device=args.device)
        
    return model