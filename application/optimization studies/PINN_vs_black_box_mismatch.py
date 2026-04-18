# if __name__ == "__main__":
#     print("THIS IS WITH SURROGATE:")
#     print("success:", res["stats"]["success"], res["stats"]["return_status"])
#     print("totals:", res["totals"])
#     print("u* well1:", np.array(res["per_well"][0]["u"]).squeeze())
#     print("y* well1:", np.array(res["per_well"][0]["y"]).squeeze())
#     # print_z_grouped(res["per_well"][0]["out"], Z_NAMES_SUR)
#
#     res = optimize_field_production(
#         model_type="surrogate",
#         N=1,
#         y_guess_list=[[3285.42, 300.822, 6910.91]],
#         u_guess_list=[[1.00, 1.00]],
#         z_guess_list=None,
#         P_max_tb_b_bar=120,
#         P_min_bh_bar=85,
#     )
#
#     res = optimize_field_production(
#         model_type="rigorous",
#         N=1,
#         y_guess_list=[[3679.08033973,
#                        289.73390193,
#                        3167.56224658,
#                        1041.96126532,
#                        50.46858403,
#                        759.52720527,
#                        249.84447542]],
#         u_guess_list=[[1.00, 1.00]],
#         z_guess_list=[[8.75897957e+06,
#                        8.42155186e+06,
#                        2.17230613e+01,
#                        2.17230613e+01]],  # [P_tb_b, P_bh, w_res] initial guess
#         P_max_tb_b_bar=130,
#         P_min_bh_bar=120,
#     )
#
