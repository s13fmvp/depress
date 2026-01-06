# MAE模型重建效果可视化函数
import matplotlib.pyplot as plt



def recon_plot(model, x, n):
    
    # spec重建效果
    _, recon_x = model.mae_spec(x)
    x = x.squeeze().detach().cpu()
    recon_x = recon_x.squeeze().detach().cpu()
    plt.figure(figsize=(30,10))
    plt.subplot(611)
    plt.pcolor(x[0,:,:])
    plt.colorbar()
    plt.subplot(612)
    plt.pcolor(recon_x[0,:,:])
    plt.colorbar()

    plt.subplot(613)
    plt.pcolor(x[1,:,:])
    plt.colorbar()
    plt.subplot(614)
    plt.pcolor(recon_x[1,:,:])
    plt.colorbar()

    plt.subplot(615)
    plt.pcolor(x[2,:,:])
    plt.colorbar()
    plt.subplot(616)
    plt.pcolor(recon_x[2,:,:])
    plt.colorbar()

    plt.savefig(n + "_recon_spec.png", dpi=300)
    plt.close()





# def recon_plot(model, x, r, p, n):
    
#     # HRRP重建效果
#     _, recon_x = model.mae_hrrp(x)
#     plt.figure()
#     plt.subplot(211)
#     plt.pcolor(x.squeeze().detach().cpu(), vmin=0, vmax=3)
#     plt.colorbar()
#     plt.subplot(212)
#     plt.pcolor(recon_x.squeeze().detach().cpu(), vmin=0, vmax=3)
#     plt.colorbar()
#     plt.savefig(n + "_recon_hrrp.png", dpi=300)
#     plt.close()


#     # PPG重建效果
#     _, recon_p = model.mae_ppg(p)
#     plt.figure()
#     plt.subplot(211)
#     plt.plot(p.squeeze().detach().cpu())
#     plt.ylim(-10,10)
#     plt.subplot(212)
#     plt.plot(recon_p.squeeze().detach().cpu())
#     plt.ylim(-10,10)
#     plt.savefig(n + "_recon_ppg.png", dpi=300)
#     plt.close()


#     # RRI重建效果
#     _, recon_r = model.mae_rri(r)
#     plt.figure()
#     plt.subplot(211)
#     plt.pcolor(r.squeeze().detach().cpu(), vmin=-0.5, vmax=0.5)
#     plt.colorbar()
#     plt.subplot(212)
#     plt.pcolor(recon_r.squeeze().detach().cpu(), vmin=-0.5, vmax=0.5)
#     plt.colorbar()
#     plt.savefig(n + "_recon_rri.png", dpi=300)
#     plt.close()


